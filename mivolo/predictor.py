from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import tqdm
from mivolo.data.misc import prepare_classification_images
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import torch
from collections import namedtuple


class Predictor:
    def __init__(self, config, verbose: bool = False):
        if config.detector_weights:
            self.detector = Detector(
                config.detector_weights, config.device, verbose=verbose
            )
        else:
            self.detector = None

        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=config.half,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def recognize(
        self, image: np.ndarray
    ) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im

    def recognize_video(self, source: str) -> Generator:
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm.tqdm(range(total_frames)):
            ret, frame = video_capture.read()
            if not ret:
                break

            detected_objects: PersonAndFaceResult = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)

            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            # add tr_persons and tr_faces to history
            for guid, data in cur_persons.items():
                # not useful for tracking :)
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            detected_objects.set_tracked_age_gender(detected_objects_history)
            if self.draw:
                frame = detected_objects.plot()
            yield detected_objects_history, frame

    def prepare_input(self, img_x):
        detected_bboxes = self.detector.predict(img_x)
        if (
            (detected_bboxes.n_objects == 0)
            or (
                not self.age_gender_model.meta.use_persons
                and detected_bboxes.n_faces == 0
            )
            or (
                self.age_gender_model.meta.disable_faces
                and detected_bboxes.n_persons == 0
            )
        ):
            # nothing to process
            print("Nothing detected. Nothing to process")
            return

        (
            faces_input,
            person_input,
            faces_inds,
            bodies_inds,
        ) = self.age_gender_model.prepare_crops(img_x, detected_bboxes)
        return faces_input, person_input

    def prepare_input_nobbox(self, imgs: List[np.ndarray]):
        assert type(imgs) == list, "imgs should be a list of images"
        return prepare_classification_images(
            imgs,
            mean=self.age_gender_model.data_config["mean"],
            std=self.age_gender_model.data_config["std"],
        )

    def inference_grads(
        self, faces_input: torch.Tensor, person_input: torch.Tensor = None
    ):
        if faces_input is None:
            print("No face found. Nothing to process")
            return

        if self.age_gender_model.meta.with_persons_model and person_input is None:
            print("No person found. Nothing to process")
            return

        if (
            self.age_gender_model.meta.with_persons_model
        ):  # Disabled for face only model
            model_input = torch.cat((faces_input, person_input), dim=1)
        else:
            model_input = faces_input

        model_input.requires_grad_()

        if self.age_gender_model.half:
            model_input = model_input.half()

        output = self.age_gender_model.model(model_input)

        age = (
            output
            * (self.age_gender_model.meta.max_age - self.age_gender_model.meta.min_age)
            + self.age_gender_model.meta.avg_age
        )

        return age

    def denormalize(self, x):
        # Image is in BGR Format
        device = x.device
        mean = (
            torch.Tensor(self.age_gender_model.data_config["mean"])
            .view(1, 3, 1, 1)
            .to(device)
        )
        std = (
            torch.Tensor(self.age_gender_model.data_config["std"])
            .view(1, 3, 1, 1)
            .to(device)
        )
        x_denormed = x * std + mean
        x_denormed = x_denormed.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

        # Convert to RGB
        for i in range(x_denormed.shape[0]):
            x_denormed[i] = cv2.cvtColor(x_denormed[i], cv2.COLOR_BGR2RGB)

        return x_denormed


def get_predictor(
    checkpoint: str,
    detector_weights: Union[str, None] = None,
    with_persons: bool = False,
    disable_faces: bool = False,
    device: str = "cuda",
    draw: bool = True,
    half: bool = False,
):
    ConfigTuple = namedtuple(
        "ConfigTuple",
        [
            "detector_weights",
            "checkpoint",
            "with_persons",
            "disable_faces",
            "device",
            "draw",
            "half",
        ],
    )

    config = ConfigTuple(
        detector_weights, checkpoint, with_persons, disable_faces, device, draw, half
    )

    predictor = Predictor(config=config, verbose=False)
    return predictor
