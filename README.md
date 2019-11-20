# Facebook chat bot
It has to be set up according to official documentation [Facebook for Developers](https://developers.facebook.com/docs/messenger-platform/).

# Heroku cloud
Heroku cloud server has to be set up according to official documentation [Getting Started on Heroku](https://devcenter.heroku.com/start).
We provide `app.js` script that we run on Heroku to connect to Facebook server. Next, this script pushes user requests into AMQP queue.

# To start demo execute this:
This script reads AMQP data and executes the trained reasoning model:

```bash
python3 rpc_server.py --expName realFridgr_FeatCls_EmbRandom_CfgArgs0 --gpus 0 --netLength 4 -r --restoreEpoch 50 --getPreds @configs/args_inference.txt
```

# Environment variables:
Don't forget to set up queue and Facebook chat bot variables in `app.js`:
```bash
CLOUDAMQP_URL
PAGE_ACCESS_TOKEN
VERIFICATION_TOKEN
```