import pickle
import unidecode # To handle accented characters from wikipedia

cw_file = "../data/words_1w.txt"
number_cw = 30000


def save_crawled_drug_tuples_to_txt(filename, save_name):

	with open(filename, 'rb') as fp:
		tp = pickle.load(fp)

	print "Total tuples : " + str(len(tp))

	l = []

	for t in tp:
		l.append(t[0].strip().lower().replace("_", " "))
		l.append(t[1].strip().lower().replace("_", " "))

	names = []

	# Following snipet to replace accented and split at /
	for name in l:
		ns = name.split("/")
		for s in ns:
			if isinstance(s, unicode):
				s = unidecode.unidecode(s)

			names.append(s)

	print "Total names present : " +  str(len(names))

	names_set_list = list(set(names))

	print "Total Distinct names present : " + str(len(names_set_list))

	names_set_list = remove_common_words(names_set_list)

	save_file = open(save_name, 'wb')

	print "Saving names ..."

	for name in names_set_list:
		save_file.write("%s\n" % name)

	save_file.close()

def remove_common_words(names, common_words_file_name = cw_file, number_most_common = number_cw):
	
	common_names = []

	print "Removing Most Common " + str(number_most_common) + " words..."

	common_file = open(common_words_file_name, 'rb')

	for line in common_file:
		common_names.append(line.strip().lower())

	common_file.close()

	common_names = common_names[:number_most_common]

	uncommon_names = []

	for name in names:
		if name not in common_names:
			uncommon_names.append(name)

	print "Total Words left after removing common words : " + str(len(uncommon_names))

	return uncommon_names

if __name__ == "__main__":
    save_crawled_drug_tuples_to_txt("../data/drug_tuples2.pkl", "../data/drug_names_wiki.txt")