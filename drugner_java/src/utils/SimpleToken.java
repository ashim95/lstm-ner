package utils;

import java.util.ArrayList;
import java.util.List;

public class SimpleToken {
	
	private String text;
	
	private List<Integer> features;
	
	private String posTag;

	
	public SimpleToken(String text) {
		this.text = text;
		this.features = new ArrayList<Integer>();
	}

	public SimpleToken(String text, List<Integer> features) {
		this.text = text;
		this.features = features;
	}

	public String getText() {
		return text;
	}

	public List<Integer> getFeatures() {
		return features;
	}

	public void setText(String text) {
		this.text = text;
	}

	public void setFeatures(List<Integer> features) {
		this.features = features;
	}
	
	public String getPosTag() {
		return posTag;
	}

	public void setPosTag(String posTag) {
		this.posTag = posTag;
	}

	public void addFeatureValue(Integer featureValue){
		this.features.add(featureValue);
	}
	
	public void addFeatureVector(List<Integer> featureVector){
		if (featureVector.size() == 0) return;
		
		for(Integer fv: featureVector){
			this.features.add(fv);
		}
	}
	
	public void addFeatureVector(int[] featureVector){
		if (featureVector.length == 0) return;
		
		for(int fv: featureVector){
			this.features.add(fv);
		}
	}
}
