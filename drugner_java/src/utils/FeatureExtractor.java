package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import dictionary.DictionaryTagger;
import dictionary.Mention;
import dictionary.Sentence;
import dictionary.Token;

public class FeatureExtractor {

	public static void main(String[] args) throws Exception {

		String input = args[0];
		String output = args[1];
		String path = args[2]; // For Relative path to drugner_java folder
		String dictionaryFilename1 = null;
		String dictionaryFilename2 = null;

		if (args.length > 3) {
			dictionaryFilename1 = args[3];
		}
		if (args.length > 4) {
			dictionaryFilename2 = args[4];
		}
		
		
		// String dictionaryFilename = "../data/drug_names_wiki.txt";
		// String input = "../logs/input/in.txt";
		// String output = "../logs/output/out.txt";

		List<SimpleSentence> simpleSentences = readData(input);

		System.out.println(simpleSentences.size());

		if (dictionaryFilename1 != null) {
			List<Sentence> sentences = convertSimpleSentencetoSentence(simpleSentences);
			sentences = dictionaryTag(sentences, dictionaryFilename1);
			simpleSentences = dictionaryTagstoFeatureVector(sentences, simpleSentences);
		}
		
		if (dictionaryFilename2 != null) {
			List<Sentence> sentences2 = convertSimpleSentencetoSentence(simpleSentences);
			sentences2 = dictionaryTag(sentences2, dictionaryFilename2);
			simpleSentences = dictionaryTagstoFeatureVector(sentences2, simpleSentences);
		}
		
		simpleSentences = OpenNLPChunker.getChunks(simpleSentences, path);

		// for(SimpleSentence sent: simpleSentences){
		// System.out.println("\n");
		// for(SimpleToken token : sent.getTokensList()){
		// System.out.println(token.getText() + token.getFeatures());
		// }
		// }

		writeFeatureVectors(simpleSentences, output);

	}

	public static void writeFeatureVectors(List<SimpleSentence> sentences, String filename) throws IOException {

		System.out.println("Writing feature vectors to file ...");

		BufferedWriter out = new BufferedWriter(new FileWriter(filename));
		String separator = "\t";
		for (SimpleSentence sent : sentences) {
			for (SimpleToken token : sent.getTokensList()) {
				String writeLine = "";
				writeLine = writeLine + token.getText() + separator;
				String features = token.getFeatures().stream().map(Object::toString)
						.collect(Collectors.joining(separator)).toString();
				writeLine = writeLine + features;
				out.write(writeLine + "\n");
			}
			out.write("\n");
		}
		out.close();
		System.out.println("Done Writing feature vectors to file");
	}

	public static List<Sentence> convertSimpleSentencetoSentence(List<SimpleSentence> simpleSentences) {

		List<Sentence> sentences = new ArrayList<Sentence>();

		for (SimpleSentence ss : simpleSentences) {

			String text = String.join(" ", ss.getTokensListAsString());
			Sentence sen = new Sentence("", "", text);

			List<Token> tokens = new ArrayList<Token>();
			int i = 0;
			int delimiter = 1;

			for (String tokenStr : ss.getTokensListAsString()) {
				int start = i;
				int end = start + tokenStr.length();
				Token token = new Token(sen, start, end);
				sen.addToken(token);
				i = end + delimiter;

			}

			sentences.add(sen);

		}
		return sentences;
	}

	public static List<Sentence> dictionaryTag(List<Sentence> sentences, String dictionaryFilename) throws IOException {

		DictionaryTagger tagger = new DictionaryTagger();
		tagger.configure();
		tagger.load(dictionaryFilename);

		for (Sentence sent : sentences) {
			tagger.tag(sent);
		}

		return sentences;
	}

	public static List<SimpleSentence> dictionaryTagstoFeatureVector(List<Sentence> sentences,
			List<SimpleSentence> simpleSentences) throws Exception {

		Integer zeroLabel = 0;
		Integer oneLabel = 1;
		Integer twoLabel = 2;
		if (sentences.size() != simpleSentences.size())
			throw new Exception("Number of sentences not same !!");
		for (int i = 0; i < sentences.size(); i++) {

			Sentence s = sentences.get(i);
			SimpleSentence ss = simpleSentences.get(i);
			if (s.getTokens().size() != ss.getTokensList().size())
				throw new Exception("Number of Tokens not same in sentence " + i + " !!");
			Map<Integer, Integer> labels = new HashMap<Integer, Integer>();

			for (int index = 0; index < ss.getTokensList().size(); index++) {
				labels.put(index, zeroLabel);
			}
			for (int index = 0; index < s.getMentions().size(); index++) {
				Mention men = s.getMentions().get(index);
				int start = men.getStart();
				int end = men.getEnd();
				if (end - start <= 1) {
					labels.put(start, oneLabel);
				} else {
					labels.put(start, oneLabel);
					for (int j = 1; j < (end - start); j++) {
						labels.put(start + j, twoLabel);
					}
				}
			}

			for (int index = 0; index < ss.getTokensList().size(); index++) {
				ss.getToken(index).addFeatureValue(labels.get(index));
			}

		}
		return simpleSentences;
	}

	public static List<SimpleSentence> readData(String input) {

		File file = new File(input);

		List<SimpleSentence> sentences = new ArrayList<SimpleSentence>();

		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));

			String line = br.readLine();
			SimpleSentence sentence = new SimpleSentence();
			while (line != null) {
				// System.out.println(line);
				if ("".equals(line.trim())) {
					if (sentence.getTokensList().size() > 0) {
						sentences.add(sentence);
						sentence = new SimpleSentence();
					}
					line = br.readLine();
					continue;
				}
				sentence.addToken(line.trim());
				line = br.readLine();
			}
			if (sentence.getTokensList().size() > 0)
				sentences.add(sentence);
		} catch (Exception e) {
			System.out.println(e);
		}

		return sentences;

	}

}
