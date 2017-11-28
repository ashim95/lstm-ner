package utils;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.List;

import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.uima.chunker.Chunker;
import opennlp.uima.postag.POSTagger;

public class OpenNLPChunker {
	
	public static List<SimpleSentence> getChunks(List<SimpleSentence> sentences, String path){
		
		InputStream chunkModelIn = null;
		ChunkerModel chunkModel = null;
		
		InputStream posModelIn = null;
		POSModel posModel = null;
		
		try{
			
			posModelIn = new FileInputStream(path + "models/chunker/opennlp-en-pos-maxent.bin");
			posModel = new POSModel(posModelIn);
			
			chunkModelIn = new FileInputStream(path + "models/chunker/opennlp-en-chunker.bin");
			chunkModel = new ChunkerModel(chunkModelIn);
			
		}
		catch (Exception e) {
			// TODO: handle exception
			System.out.println(e);
		}
		System.out.println("Loaded the OpenNLP Chunker ");
		
		POSTaggerME posTagger = new POSTaggerME(posModel);
		ChunkerME chunker = new ChunkerME(chunkModel);
		
		System.out.println("Finding chunks for sentences ...");
		
		for (SimpleSentence sent : sentences){
			String toks[] = sent.getTokensListAsString().toArray(new String[0]);
			
			String posTags[] = posTagger.tag(toks);
			
			String chunks[] = chunker.chunk(toks, posTags);
			
			for(int index=0;index<toks.length;index++){
				if (chunks[index].contains("B-NP")){
					sent.getToken(index).addFeatureValue(1);
				}
				else if (chunks[index].contains("I-NP")) {
					sent.getToken(index).addFeatureValue(2);
				}
				else{
					sent.getToken(index).addFeatureValue(0);
				}
			}
		}
		
		return sentences;
	}
}
