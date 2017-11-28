import nltk
import subprocess

feature_on_val = 1.0
feature_off_val = 0.0


def extract_features(list_of_sentences, config):

	features = []

	if config.use_hand_crafted:
		features, is_size_same = get_features_from_java_program(list_of_sentences, config)
		if is_size_same:
			return features
		else:
			features = []


	for sent in list_of_sentences:
		sent_features = []
		for word in sent:
			feat = []

			# if config.use_dictionary:
			# 	if word.strip().lower() in config.gazetteer:
			# 		feat.append( feature_on_val )
			# 	else:
			# 		feat.append( feature_off_val )



			sent_features.append(feat)

		features.append(sent_features)

	return features


def get_features_from_java_program(list_of_sentences, config):

	print "Now starting processing of java program ..."
	write_sentences_to_file(list_of_sentences, config.java_input_path)

	command = "java -Xmx32g -cp drugner_java/jars/drugner_java_1.0.jar:drugner_java/lib/opennlp-tools-1.8.3.jar" 
	command = command + " utils.FeatureExtractor 'logs/input/in.txt' 'logs/output/out.txt' 'drugner_java/' 'data/drug_names_wiki.txt'"

	subprocess.call(command, shell=True)

	print "Executed java program successfully "

	features = read_features_from_file(config.java_output_path, config)

	is_size_same = check_size(list_of_sentences, features)

	print "Size check performed successfully "

	return features, is_size_same


def write_sentences_to_file(list_of_sentences, filename):

	fp = open(filename, "wb")
	for sent in list_of_sentences:
		for word in sent:
			fp.write(word.strip() + "\n")
		fp.write("\n")
	fp.close()

def read_features_from_file(filename, config):
	features = []
	num = config.features_index
	fp = open(filename, 'rb')
	sent = []
	for line in fp:
		line = line.strip()
		if line == "":
			if len(sent) != 0:
				# print sent
				features.append(sent)
			sent = []
			continue
		line_contents = line.split("\t")
		feats = []
		line_contents = line_contents[1:]
		line_contents = [line_contents[i] for i in num]
		for lc in line_contents:
			feats.append(float(lc))
		# print feats
		sent.append(feats)

	return features

def check_size(list_of_sentences, features):
	if len(list_of_sentences) != len(features):
		print "Size of sentences not same"
		print "Total Sentences : " + str(len(list_of_sentences))
		print "Features only present for : " + str(len(features))

		return False

	for i in range(len(list_of_sentences)):
		sent = list_of_sentences[i]
		feats = features[i]

		if len(sent) != len(feats):
			print "No of words not same for sentence number: " + str(i)
			print "Total Words in sentence : " + str(len(sent))
			print "Features only present for : " + str(len(feats))
			print sent
			print feats
			return False
	return True