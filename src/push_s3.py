import boto3
import os


# ACCESS_KEY = "AKIAW3CEOBGF6VFBIC7S"
# SECRET_KEY = "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"
bucket_name = "toxic-comments19032023"
session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
    aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
)


def upload_file(session, file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = session.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except:
        print(f"Couldn't upload the file")
        return "Failed"

    return "Upload success"


print(
    upload_file(
        session,
        "../model/model-2023-04-21-18-12-05-017588.bin",
        bucket_name,
        "best_weights",
    )
)
