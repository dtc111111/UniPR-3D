import pickle
import numpy as np
import redis
import socket
import struct
import time
import sys
import io
from PIL import Image
import cv2



# An image wrapper that caches image data in memory for faster access
class CacheImage:
    def __init__(self, cache_size=30000):
        self.cache_size = cache_size
        self.cache = {}


    def get_image(self, path, load_image_func):
        if path in self.cache:
            print("Cache hit for:", path)
            return self.cache[path]
        else:
            image = load_image_func(path)
            if len(self.cache) < self.cache_size:
                self.cache[path] = image
            return image

# Cache implementation using Redis (requires redis-py and a running Redis server)
class Mat_Redis_Utils():
    def __init__(self, host="rdb", port=6379, db=0, retry_delay=5):
        try:
            socket.gethostbyname(host)
        except socket.error:
            host = "localhost"
        self.host = host
        self.port = port
        self.db = db
        self.retry_delay = retry_delay
        self.handle = redis.Redis(host, port, db)
        self.dtype_table = [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64
        ]

    def connect_to_redis(self):
        while True:
            try:
                self.handle = redis.Redis(self.host, self.port, self.db)
                self.handle.ping()  # Try pinging Redis to ensure connectivity
                break  # connection established, break from the loop
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, ConnectionResetError, redis.exceptions.BusyLoadingError) as e:
                time.sleep(self.retry_delay)  # wait for some time before next retry
            except KeyboardInterrupt:
                print("User interruption. Quitting...")
                sys.exit(0)

    def robust_redis_command(self, command_func, *args, **kwargs):
        while True:
            try:
                return command_func(*args, **kwargs)
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, ConnectionResetError, redis.exceptions.BusyLoadingError) as e:
                print(f"Redis connection lost. Retrying connecting..., press Ctrl+C to quit")
                self.connect_to_redis()  # try to reconnect
            except KeyboardInterrupt:
                print("User interruption. Quitting...")
                sys.exit(0)


    def mat_to_bytes(self, arr):
        dtype_id = self.dtype_table.index(arr.dtype)
        header = struct.pack('>'+'I' * (2+arr.ndim),
                             dtype_id, arr.ndim, *arr.shape)
        data = header + arr.tobytes()
        return data

    def bytes_to_mat(self, data):
        dtype_id, ndim = struct.unpack('>II', data[:8])
        dtype = self.dtype_table[dtype_id]
        shape = struct.unpack('>'+'I'*ndim, data[8:4*(2+ndim)])
        arr = np.frombuffer(data[4*(2+ndim):], dtype=dtype, offset=0)
        arr = arr.reshape((shape))
        return arr

    def set(self, key, arr):
        return self.robust_redis_command(self.handle.set, key, self.mat_to_bytes(arr))

    def get(self, key, dtype=np.float32):
        data = self.robust_redis_command(self.handle.get, key)
        if data is None:
            raise ValueError('%s not exist in Redis' % (key))
        return self.bytes_to_mat(data)

    def set_PIL(self, key):
        data = open(key, "rb").read()
        image = Image.open(io.BytesIO(data))
        self.robust_redis_command(self.handle.set, key, data)
        return image

    def get_PIL(self, key):
        data = self.robust_redis_command(self.handle.get, key)
        if data is None:
            raise ValueError('%s not exist in Redis' % (key))
        return Image.open(io.BytesIO(data))

    def set_cv2(self, key):
        with open(key, "rb") as f:
            data = f.read()
        self.robust_redis_command(self.handle.set, key, data)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        return image

    def get_cv2(self, key):
        data = self.robust_redis_command(self.handle.get, key)
        if data is None:
            raise ValueError('%s not exist in Redis' % (key))
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        return image

    def load_PIL(self, key):
        if not self.robust_redis_command(self.handle.execute_command, 'EXISTS', key):
            return self.set_PIL(key)
        else:
            # print("PIL Cache hit for:", key)
            return self.get_PIL(key)
    
    def load_cv2(self, key):
        if not self.robust_redis_command(self.handle.execute_command, 'EXISTS', key):
            return self.set_cv2(key)
        else:
            return self.get_cv2(key)

    def exists(self, key):
        return bool(self.robust_redis_command(self.handle.execute_command, 'EXISTS', key))

    def ls_keys(self):
        return self.robust_redis_command(self.handle.execute_command, 'KEYS *')

    def flush_all(self):
        print('Del all keys in Redis')
        return self.robust_redis_command(self.handle.execute_command, 'flushall')