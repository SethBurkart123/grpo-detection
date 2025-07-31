from src.detection import detect

string = "The jade gift represents a cultural inheritance that remained largely untouched throughout her upbringing. For second-generation immigrants, this distance from ancestral traditions is a common experience, creating feelings of alienation from a heritage that rightfully should belong to them."

# ai detection for a reward function
#print(detect("Help me! I'm not sure what is happening! How are you today? What is going on guys!!! I don't understand?", 'main.log'))
print(detect(string, 'main.log'))
