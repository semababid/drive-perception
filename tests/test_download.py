"""Fast checks for the KITTI download module. The actual fetch needs network and is
exercised by running the script, not by CI, so nothing here hits the wire."""

from drive_perception.data import download


def test_mirror_is_public_kitti_s3():
    # The whole no-registration story rests on using the public S3 mirror, so guard it.
    assert download.IMAGES_URL.startswith("https://s3.eu-central-1.amazonaws.com/avg-kitti")
    assert download.LABELS_URL.endswith("data_object_label_2.zip")


def test_only_training_split_is_targeted():
    # testing/ has no public labels, so both prefixes must point at the training split.
    assert download.IMAGE_PREFIX == "training/image_2/"
    assert download.LABEL_PREFIX == "training/label_2/"
