package utils;

import java.util.ArrayList;
import java.util.List;

public class SimpleToken {
	
	private String text;
	
	private List<Integer> features;

	
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
	
	public void addFeatureValue(Integer featureValue){
		this.features.add(featureValue);
	}
}
