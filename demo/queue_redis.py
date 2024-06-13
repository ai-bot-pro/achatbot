import redis

redis_client = redis.Redis(
    host='redis-12259.c240.us-east-1-3.ec2.redns.redis-cloud.com',
    port=12259,
    password='7uywUnJtUa22Jpt2MMZOCGDwoseoqj6z')

# Test connection
res = redis_client.ping()
print(res)
# Clear Redis database (optional)
redis_client.flushdb()
