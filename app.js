var express = require("express");
var request = require("request");
var bodyParser = require("body-parser");
const amqp = require('amqplib');
const EventEmitter = require('events');
const uuid = require('uuid');

const RABBITMQ = process.env.CLOUDAMQP_URL;

// pseudo-queue for direct reply-to
const REPLY_QUEUE = 'amq.rabbitmq.reply-to';
const q = 'rpc_queue';

const createClient = rabbitmqconn =>
  amqp
    .connect(rabbitmqconn)
    .then(conn => conn.createChannel())
    .then(channel => {
      channel.responseEmitter = new EventEmitter();
      channel.responseEmitter.setMaxListeners(0);
      channel.consume(
        REPLY_QUEUE,
        msg => {
          channel.responseEmitter.emit(
            msg.properties.correlationId,
            msg.content.toString('utf8'),
          );
        },
        { noAck: true },
      );
      return channel;
    });

const sendRPCMessage = (channel, message, rpcQueue) =>
  new Promise(resolve => {
    const correlationId = uuid.v4();
    channel.responseEmitter.once(correlationId, resolve);
    channel.sendToQueue(rpcQueue, Buffer.from(message), {
      correlationId,
      replyTo: REPLY_QUEUE,
    });
  });

var channel = null;
const init = async () => {
  channel = await createClient(RABBITMQ);
  
  var i;
  for (i = 0; i < 10; i++) {
    var message = uuid.v4();
    console.log(`[ ${new Date()} ] Message sent: ${JSON.stringify(message)}`);
    var response = await sendRPCMessage(channel, message, q);
    console.log(`[ ${new Date()} ] Message received: ${response}`);
  }

  //return channel;
};

try {
    init();
  } catch (e) {
    console.log(e);
  }

//////////////////////////////////////////////
var app = express();
app.use(bodyParser.urlencoded({extended: false}));
app.use(bodyParser.json());
app.listen((process.env.PORT || 5000));

// Server index page
app.get("/", function (req, res) {
    res.send("Deployed!");
});

// Used for verification
//app.get("/webhook", function (req, res) {
//    if (req.query["hub.verify_token"] === "this_is_my_token") {
//      console.log("Verified webhook");
//      res.status(200).send(req.query["hub.challenge"]);
//    } else {
//      console.error("Verification failed. The tokens do not match.");
//      res.sendStatus(403);
//    }
//  });
// Facebook Webhook
app.get("/webhook", function (req, res) {
    if (req.query["hub.verify_token"] === process.env.VERIFICATION_TOKEN) {
        console.log("Verified webhook");
        res.status(200).send(req.query["hub.challenge"]);
    } else {
        console.error("Verification failed. The tokens do not match.");
        res.sendStatus(403);
    }
});

// All callbacks for Messenger will be POST-ed here
app.post("/webhook", function (req, res) {
    // Make sure this is a page subscription
    if (req.body.object == "page") {
        // Iterate over each entry
        // There may be multiple entries if batched
        req.body.entry.forEach(function(entry) {
            // Iterate over each messaging event
            entry.messaging.forEach(function(event) {
                if (event.postback) {
                    processPostback(event);
                } else if (event.message) {
                    processMessage(event);
                }
            });
        });

        res.sendStatus(200);
    }
});

function processPostback(event) {
    //
    var senderId = event.sender.id;
    var payload = event.postback.payload;

    if (payload === "Greeting") {
        // Get user's first name from the User Profile API
        // and include it in the greeting
        request({
            url: "https://graph.facebook.com/v2.6/" + senderId,
            qs: {
                access_token: process.env.PAGE_ACCESS_TOKEN,
                fields: "first_name"
            },
            method: "GET"
        }, function(error, response, body) {
            var greeting = "";
            if (error) {
                console.log("Error getting user's name: " + error);
            } else {
                var bodyObj = JSON.parse(body);
                name = bodyObj.first_name;
                greeting = "Hi " + name + ". ";
            }
            var message = greeting + "My name is Beta's Fridge Bot. I can tell you about my contents. What would you like to know about?";
            sendMessage(senderId, {text: message});
        });
    }
}

async function processMessage(event) {
    if (!event.message.is_echo) {
        var message = event.message;
        var senderId = event.sender.id;

        console.log("Received message from senderId: " + senderId);
        console.log("Message is: " + JSON.stringify(message));
        // You may get a text or attachment but not both
        if (message.text) {

            ////////////////////////////////////////////////////////////////
            const question = message.text.toLowerCase().trim();
            const response = await sendRPCMessage(channel, question, q);
            sendMessage(senderId, {text: response});
            //obj = JSON.parse(response);
            //sendMessage(senderId, {text: obj.answer});
            /*const response = await sendRPCMessage(channel, question, q);
            obj = JSON.parse(response);
            message = {
                attachment: {
                    type: "template",
                    payload: {
                        template_type: "generic",
                        elements: [{
                            title: movieObj.Title,
                            subtitle: "Is this the movie you are looking for?",
                            image_url: movieObj.Poster === "N/A" ? "http://placehold.it/350x150" : movieObj.Poster,
                            buttons: [{
                                type: "postback",
                                title: "Yes",
                                payload: "Correct"
                            }, {
                                type: "postback",
                                title: "No",
                                payload: "Incorrect"
                            }]
                        }]
                    }
                }
            };
            sendMessage(senderId, {text: obj.answer});*/
            ////////////////////////////////////////////////////////////////

        } else if (message.attachments) {
            sendMessage(senderId, {text: "Sorry, I don't understand your request."});
        }
    }
}

// sends message to user
function sendMessage(recipientId, message) {
    request({
        url: "https://graph.facebook.com/v2.6/me/messages",
        qs: {access_token: process.env.PAGE_ACCESS_TOKEN},
        method: "POST",
        json: {
            recipient: {id: recipientId},
            message: message,
        }
    }, function(error, response, body) {
        if (error) {
            console.log("Error sending message: " + response.error);
        }
    });
}
