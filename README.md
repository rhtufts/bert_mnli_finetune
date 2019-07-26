# Bert MNLI Finetuning
This repo implements a fine-tuning of the BERT-base model from [BERT](https://arxiv.org/abs/1810.04805) on the [MNLI](https://www.nyu.edu/projects/bowman/multinli/).

The model achieves 0.843 combined accuracy (on mismatched and matched pairs) on the test set of MNLI, which is comparable with the "Stacking BERT-base" [paper](http://proceedings.mlr.press/v97/gong19a/gong19a.pdf) ,currently 15th on the GLUE leaderboard.
That puts this implementation almost 10% points higher than the BiLSTM + ELMo + Attention baseline of the GLUE dataset.


## Usage

Example request:
>curl -H 'Content-type: application/json' -X POST --data '{"pairs":[{ "text_a":"Eyewitnesses said de Menezes had jumped over the turnstile at Stockwell subway station.","text_b":"The documents leaked to ITV News suggest that Menezes walked casually into the subway station."}]}' http://ec2-54-171-220-199.eu-west-1.compute.amazonaws.com:8888/nli

NOTE: The "pairs" key accepts a json array of dictionaries, allowing mulptiple pairs to be sent at once.
This was done to allow for batched inference, but since the current deployment is on a single CPU with no GPU, this is unlikely to lead to any speed improvement.

Example response:
>{
   "predictions":[
      {
         "input_pair":{
            "text_a":"Eyewitnesses said de Menezes had jumped over the turnstile at Stockwell subway station.",
            "text_b":"The documents leaked to ITV News suggest that Menezes walked casually into the subway station."
         },
         "label":"contradiction",
         "confidence":[
            0.9913362264633179,
            0.00043108518002554774,
            0.008232601918280125
         ]
      }
   ]
}

The "confidence" attribute corresponds to the "probability" vector the model produces over the categories,
where (1,0,0) = "Contradiction, (0,1,0) = "Neutral", and (0,0,1) = "Entailment".
