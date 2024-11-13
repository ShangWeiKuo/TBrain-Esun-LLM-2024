import configparser

configParser = configparser.RawConfigParser()
configFilePath = "config/model_name.ini"
configParser.read(configFilePath)

tokenizer_name = configParser.get("Model", "tokenizer_name")
model_name = configParser.get("Model", "model_name")
