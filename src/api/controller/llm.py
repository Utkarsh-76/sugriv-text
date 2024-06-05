
from src.db.helpers.redis import RedisDB
from src.utils.logger import logging as logger
from src.api.vo.llm_request import LLMRequest
from src.llm.model.sugriv import sugriv
from fastapi.responses import JSONResponse
from collections import defaultdict

redis = RedisDB()
sugriv = sugriv.get_model()

def generate_top_k_results(request:LLMRequest):
    ''' generates the top k results using a greedy sampling methodology '''
    try:
        k_nearest = redis.get_top_k(request.prompt,int(request.top_k))
        k_nearest = []
        if k_nearest != []:
            logger.debug('got the completions from redis cache')
            results  = k_nearest
        else:
            logger.debug('getting the top k completions from the llm')
            results = sugriv.top_k(request.prompt,int(request.top_k))
            logger.info('got the completions from llm')
            logger.info('saving the result in redis db ')

            response = []

            for result in results:
                response.append({result[0]:result[1]})
                redis.put(request.prompt,result[0],result[1])

            merged_dict = {k: v for d in response for k, v in d.items()}

        return JSONResponse(content={"completion": merged_dict },status_code=200)
    except RuntimeError as e:
        logger.error({'error':'run time error occured'})
        logger.error(e)
        return {'error':'run time error occured'}