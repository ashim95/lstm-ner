package utils;



import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.HierarchicalConfiguration;
import org.apache.commons.configuration.XMLConfiguration;

import banner.eval.BANNER;
import banner.postprocessing.PostProcessor;
import banner.tagging.CRFTagger;
import banner.tokenization.Tokenizer;
import banner.types.Mention;
import banner.types.Sentence;
import banner.types.Mention.MentionType;
import dnorm.core.DiseaseNameAnalyzer;
import dnorm.core.Lexicon;
import dnorm.core.MEDICLexiconLoader;
import dnorm.core.SynonymTrainer;
import dnorm.types.FullRankSynonymMatrix;
import dragon.nlp.tool.Tagger;
import dragon.nlp.tool.lemmatiser.EngLemmatiser;
import dragon.util.EnvVariable;

public class DNormDiseaseTagger {
	
	private static final String PROPERTIES_FILE = "resources/application.properties";
	private static String configurationFilename;
	private static String lexiconFilename;
	private static String matrixFilename;
//	private static SynonymTrainer syn;
	private static CRFTagger tagger;
	private static Tokenizer tokenizer;
	private static PostProcessor postProcessor;
	
	
	
	public static List<SimpleSentence> getDiseaseTags(List<SimpleSentence> sentences, String path) throws ConfigurationException, IOException {
		System.out.println("Using DNorm Disease Tagger ...");
		loadProperties(path);
		configure();
		for(SimpleSentence sent: sentences){
			String text = "";
			String docId = "123";
			String sentId = "1";
			text = sent.getTokensListAsString().stream().collect(Collectors.joining(" ")).toString();
			banner.types.Sentence sentence1 = new banner.types.Sentence(docId + "-" + sentId, docId,text);
			banner.types.Sentence sentence2 = BANNER.process(tagger, tokenizer, postProcessor, sentence1);
			Set<String> mentionTexts = new HashSet();
			for (Mention mention : sentence2.getMentions(MentionType.Found)) {
				mentionTexts.add(mention.getText().trim().toLowerCase());
			}
			for(SimpleToken token: sent.getTokensList()){
				if (mentionTexts.contains(token.getText().trim().toLowerCase())){
					token.addFeatureValue(1);
				}
				else{
					token.addFeatureValue(0);
				}
			}
		}
		System.out.println("Completed using DNorm Disease Tagger ...");
		return sentences;
	}

	public static void configure() throws ConfigurationException, IOException{
		long start = System.currentTimeMillis();
//		DiseaseNameAnalyzer analyzer = DiseaseNameAnalyzer.getDiseaseNameAnalyzer(true, true, false, true);
//		Lexicon lex = new Lexicon(analyzer);
//		MEDICLexiconLoader loader = new MEDICLexiconLoader();
//		loader.loadLexicon(lex, lexiconFilename);
//		lex.prepare();
//		System.out.println("Lexicon loaded; elapsed = " + (System.currentTimeMillis() - start));
//		
//		
//		FullRankSynonymMatrix matrix = FullRankSynonymMatrix.load(new File(matrixFilename));
//		syn = new SynonymTrainer(lex, matrix, 1000);
//		System.out.println("Matrix loaded; elapsed = " + (System.currentTimeMillis() - start));
		System.out.println(configurationFilename);
		HierarchicalConfiguration config = new XMLConfiguration(configurationFilename);
		EnvVariable.setDragonHome(".");
		EnvVariable.setCharSet("US-ASCII");
		EngLemmatiser lemmatiser = BANNER.getLemmatiser(config);
		Tagger posTagger = BANNER.getPosTagger(config);
		HierarchicalConfiguration localConfig = config.configurationAt(BANNER.class.getPackage().getName());
		String modelFilename = localConfig.getString("modelFilename");
		
		tokenizer = BANNER.getTokenizer(config);
		postProcessor = BANNER.getPostProcessor(config);
		tagger = CRFTagger.load(new File(modelFilename), lemmatiser, posTagger);
		System.out.println("BANNER loaded; elapsed = " + (System.currentTimeMillis() - start));
		
		
	}

	public static void loadProperties(String path) {
		String pathCat = null;
		if (".".equals(path)) 
			pathCat = "";
		else
			pathCat = path;
		// Read properties
		Properties properties = new Properties();

		try {
			properties.load(new FileInputStream(new File(PROPERTIES_FILE)));
		} catch (Exception e) {
			e.printStackTrace();
		}

		String config = properties.getProperty("dNormConfigurationFilename");
		configurationFilename = pathCat + config;
		
		config = properties.getProperty("dNormLexiconFilename");
		lexiconFilename = pathCat + config;
		
		config = properties.getProperty("dNormMatrixFilename");
		matrixFilename = pathCat + config;
		
//		config = properties.getProperty("dNormLexiconFilename");
//		lexiconFilename = config;
//		
//		config = properties.getProperty("dNormLexiconFilename");
//		lexiconFilename = config;

	}
}
