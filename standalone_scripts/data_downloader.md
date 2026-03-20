# Data Downloader: SageMaker to S3

Quick reference for downloading large datasets from URLs to SageMaker and syncing to S3.

## Basic Workflow

```bash
# 1. Download with resume capability
wget <URL>
# If download fails, resume with:
wget -c <URL>

# 2. Extract the file
unzip <filename>.zip -d /home/ec2-user/SageMaker/<folder-name>
# Or for tar.gz files:
# tar -xzf <filename>.tar.gz -C /home/ec2-user/SageMaker/<folder-name>

# 3. Configure AWS CLI for better S3 sync performance
aws configure set default.s3.max_concurrent_requests 20

# 4. Sync to S3
aws s3 sync /home/ec2-user/SageMaker/<folder-name> s3://<bucket-name>/<folder-name>/
```

## Example

```bash
# Download dataset
wget https://example.com/dataset.zip
# If interrupted:
wget -c https://example.com/dataset.zip

# Extract to SageMaker directory
unzip dataset.zip -d /home/ec2-user/SageMaker/my-dataset

# Optimize S3 sync
aws configure set default.s3.max_concurrent_requests 20

# Sync to S3
aws s3 sync /home/ec2-user/SageMaker/my-dataset s3://my-bucket/my-dataset/
```

## Notes

- **SageMaker location**: `/home/ec2-user/SageMaker` (standard SageMaker notebook path)
- **EBS location**: May differ if using custom EBS mounts
- **S3 optimization**: `max_concurrent_requests 20` significantly speeds up large transfers
- **Resume downloads**: Always use `wget -c` if a download is interrupted
