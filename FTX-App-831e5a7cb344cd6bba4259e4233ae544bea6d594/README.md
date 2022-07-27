# Basic-Trading-App

Basic boilerplate app for fully automated algorithmic trading

1. Risk management logic goes into RiskManager.py
2. Exchange integrations go into Clients.py
3. Any logic related to trade execution (e.g. buying/selling) goes into Services.py

If you have an FTX account feel free to enter your API keys/secret into the config and see if it works. You should be able to run this on multiple exchanges. 

If any help is needed with the set-up feel free to contact me at xaes@pm.me

# How the application works

1. Adding your own exchange. 

Simply inherit the base exchange client object and customise it to work with your own exchange as per their API documentation. 

You may need to add your key information into the config.json file and modify the Exchange Enum object to include your exchange

2. Customising the risk management system

The Risk Manager class has a machine learning pipeline for predicting market risk ahead of time and updates the risk metrics to the config.json file. 

Currently it uses a Gradient Boosting Regressor but you can modify this pipeline to suit your own model and needs

3. Customising Trade Execution Logic

Inside Services.py you'll see separate Buy and Sell services that inherit from a Base Service. 

Inside each of these services is some basic RSI logic that will act on buying and selling depending on the service. Feel free to modify this logic here to suit your own needs. 
