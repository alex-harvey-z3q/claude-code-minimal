resource "aws_s3_bucket" "raw" {
  bucket = "${local.project}-raw-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "parsed" {
  bucket = "${local.project}-parsed-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "raw" {
  bucket = aws_s3_bucket.raw.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_versioning" "parsed" {
  bucket = aws_s3_bucket.parsed.id
  versioning_configuration { status = "Enabled" }
}
