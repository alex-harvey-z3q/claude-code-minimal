terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    dns = {
      source  = "hashicorp/dns"
      version = ">= 3.0"
    }
  }
}

provider "aws" {
  region = local.aws_region
}

provider "dns" {}
