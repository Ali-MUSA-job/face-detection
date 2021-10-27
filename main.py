"""
This script uses MTCNN library to detect human faces.

---How to use---
1) pip3 install -r requirements.txt
2) python3 main.py

Author : Ali MUSA
Date : 01/08/2021
"""

from utils import *

load_dotenv(verbose=True)
env_path = Path('.') / '{}.env'.format(sys.argv[1])
load_dotenv(dotenv_path=env_path)

log_level = os.getenv('LOG_LEVEL')
graylog_host = os.getenv('GRAYLOG_HOST')
graylog_port = os.getenv('GRAYLOG_PORT')
logger = logging.getLogger()

if graylog_host:
    graylog_handler = graypy.GELFUDPHandler(graylog_host, int(graylog_port))
    logger.addHandler(graylog_handler)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(log_level)

# force script to use GPU (to use CPU change (0) to (-1))
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def signal_handler(sig, frame):
    logger.info('You pressed Ctrl+C!')
    connection.close()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# redis connection 
redis = Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'), db=0)

# rabbitmq connection
try:
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST')))
    channel = connection.channel()
    if connection.is_open:
        logger.info('RabbitMQ Connection Complete')
except Exception as error:
    logger.error('Error:', error.__class__.__name__)
    exit(1)


def callback(ch,method,properties,body):
    """ This function to do face detection using decoded json data and publish result to rabbitmq.
        :returns : void
    """

    global redis

    json_data = json_load(body)       
    redis_data = redis_get(redis,json_data)
    if redis_data:           
        original_frame = get_frame(redis_data)   
        faces_arr = do_detection(json_data,original_frame)

        if len(faces_arr) > 0:
            for i in range(len(faces_arr)):
                x,y,w,h = faces_arr[i]["detected_img_coordinate"]

                json_data['coordinates'] = [[str(w+1), str(h+1), str(x), str(y)]]
                json_data['face_uuid'] = faces_arr[i]["face_uuid"]
                json_data['frame_h'] = faces_arr[i]["face_arr"].shape[0]
                json_data['frame_w'] = faces_arr[i]["face_arr"].shape[1]
                json_data['payload'] = {"face_success_rate":faces_arr[i]["success_rate"]}

                image = np.frombuffer(faces_arr[i]["face_arr"], dtype=np.uint8) 
                string = np.array(image).tostring()

                redis.set(faces_arr[i]["face_uuid"], string , 600)

                channel.basic_publish(
                    exchange='',
                    routing_key=os.getenv('RABBIT_OUTPUT_QUEUE'),
                    body=json.dumps(json_data),          
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                        expiration="60000",
                    ))

    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=os.getenv('RABBIT_INPUT_QUEUE'), on_message_callback=callback)
channel.start_consuming()
