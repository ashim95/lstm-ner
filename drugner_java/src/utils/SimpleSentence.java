package utils;

import java.util.ArrayList;
import java.util.List;

public class SimpleSentence {
	
	private List<SimpleToken> tokens;
	
	
	public SimpleSentence() {
		tokens = new ArrayList<SimpleToken>();
	}

	public void addToken(String token){
		tokens.add(new SimpleToken(token.trim()));
	}
	
	public void addToken(SimpleToken token){
		tokens.add(token);
	}
	
	public List<SimpleToken> getTokensList(){
		return tokens;
	}
	
	public List<String> getTokensListAsString(){
		List<String> tokenString = new ArrayList<String>();
		
		for (SimpleToken token : tokens){
			tokenString.add(token.getText());
		}
		return tokenString;
	}
	
	public SimpleToken getToken(int index){
		return tokens.get(index);
	}
}
