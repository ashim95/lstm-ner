package utils;

import java.util.List;

import dragon.nlp.Sentence;
import dragon.nlp.Word;
import dragon.nlp.tool.HeppleTagger;
import dragon.nlp.tool.Tagger;
import dragon.util.EnvVariable;

public class DragonPosTagger {

	private static Tagger posTagger;

	public static List<SimpleSentence> getPosTag(List<SimpleSentence> sentences, String path) {
		System.out.println("Using Dragon POS Tagger ...");
		configure(path);

		for (SimpleSentence sent : sentences) {
			dragon.nlp.Sentence posSentence = null;
			if (posTagger != null) {
				int size = sent.getTokensList().size();
				posSentence = new dragon.nlp.Sentence();
				for (int i = 0; i < size; i++)
					posSentence.addWord(new Word(sent.getTokensListAsString().get(i)));
				posTagger.tag(posSentence);
			}

			for (int index = 0; index < sent.getTokensList().size(); index++) {
				int posIndex = posSentence.getWord(index).getPOSIndex();
				String posTag = posSentence.getWord(index).getPOSLabel();
				int[] fv = getPosIndexAsOneHotVector(posIndex);
				sent.getToken(index).setPosTag(posTag);
				sent.getToken(index).addFeatureVector(fv);
			}
		}
		System.out.println("Completed using Dragon POS Tagger ...");
		return sentences;
	}

	private static int[] getPosIndexAsOneHotVector(int index) {
		int MAX_POS = 9;
		int[] fv = new int[MAX_POS + 1];
		fv[index] = 1;
		return fv;

	}

	private static void configure(String path) {
		EnvVariable.setDragonHome(path);
		EnvVariable.setCharSet("US-ASCII");
		String dataDir = "nlpdata/tagger";
		String taggerDataDirectory = null;

		// Using Hepple Tagger
		if (".".equals(path))
			taggerDataDirectory = dataDir;
		else
			taggerDataDirectory = path + dataDir;
		posTagger = new HeppleTagger(taggerDataDirectory);

	}

}
