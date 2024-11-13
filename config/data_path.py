import configparser

configParser = configparser.RawConfigParser()
configFilePath = "config/data_path.ini"
configParser.read(configFilePath)

source_path = configParser.get("Data", "source_path")
questions_path = configParser.get("Data", "questions_path")
output_path = configParser.get("Data", "output_path")
insurance_json_name = configParser.get("Data", "insurance_json_name")
finance_json_name = configParser.get("Data", "finance_json_name")
faq_json_name = configParser.get("Data", "faq_json_name")
