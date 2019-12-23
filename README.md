# Demo app code for
**[Smart Home Appliances: Chat with Your Fridge](https://arxiv.org/pdf/1912.09589.pdf)**

## Demo video
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/UkapMNcXlq8/0.jpg)](https://www.youtube.com/watch?v=UkapMNcXlq8)

## Facebook chatbot
Facebook app has to be set up according to the official documentation [Facebook for Developers](https://developers.facebook.com/docs/messenger-platform/).

## Heroku cloud
Heroku cloud server has to be set up according to the official documentation [Getting Started on Heroku](https://devcenter.heroku.com/start).
We provide `app.js` script that can be run on Heroku server to work with Facebook server. On the client side, this script pushes user requests into AMQP queue.

## To start demo execute command
This script reads AMQP data and executes the trained reasoning model:

```bash
CUDA_VISIBLE_DEVICES=0, python3 rpc_server.py --expName fridgr_FeatCls_EmbRandom_CfgArgs0 --gpus 0 --netLength 4 --restoreEpoch 25 --getPreds @configs/args_inference.txt
```

## Environment variables
Don't forget to set up queue and Facebook chatbot variables in `app.js`:
```bash
CLOUDAMQP_URL
PAGE_ACCESS_TOKEN
VERIFICATION_TOKEN
```
