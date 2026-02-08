locals {
  project                 = "wiki-rag-agentic"
  aws_region              = "ap-southeast-2"
  db_username             = "wikirdb"
  github_repo             = "alex-harvey-z3q/wiki-rag-agentic"
  laptop_ip               = "125.63.140.154/32"
  bedrock_chat_model_id   = "anthropic.claude-3-5-sonnet-20241022-v2:0"
  bedrock_embed_model_id  = "amazon.titan-embed-text-v2:0"
  embed_dim               = "1024"
}
