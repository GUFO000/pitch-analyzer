#!/bin/bash
# Deploy backend to Elastic Beanstalk
zip -r backend.zip backend/
aws elasticbeanstalk create-application-version \
    --application-name your-app \
    --version-label v1 \
    --source-bundle S3Bucket="your-bucket",S3Key="backend.zip"
aws elasticbeanstalk update-environment \
    --environment-name your-env \
    --version-label v1 