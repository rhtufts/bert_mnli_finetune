from flask import Flask, request
from flask_restful import Resource, Api, abort
from test_mode import *
import logging
from waitress import serve
logging.getLogger().addHandler(logging.StreamHandler())
app = Flask(__name__)
api = Api(app)

btf_tokenizer, btf = load_model_from_pretrained()

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class Predict_entailment(Resource):
    def post(self):
        raw_json = request.get_json()
        if 'pairs' not in raw_json:
            abort(404, message="invalid request")
        pairs = raw_json['pairs']
        prod_dataloader = create_dataloader_from_input(pairs, btf_tokenizer)
        predictions = predict_entailment(prod_dataloader, btf)
        logging.debug(predictions)
        predictions = zip(pairs,predictions[0], predictions[1])
        return {'predictions':[{'input_pair':p[0], 'label':p[1], 'confidence':p[2]} for p in predictions]}

api.add_resource(Predict_entailment, '/nli')
api.add_resource(HelloWorld, '/')

if __name__ == '__main__':

    # app.run(debug=False)
    serve(app, host='0.0.0.0', port=8000)