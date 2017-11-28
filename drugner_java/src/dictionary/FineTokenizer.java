package dictionary;

import java.util.ArrayList;
import java.util.List;


public class FineTokenizer {

	public FineTokenizer() {
		// Empty
	}

	private static boolean isPunctuation(char ch) {
		if (Character.isLetter(ch))
			return false;
		if (Character.isDigit(ch))
			return false;
		if (Character.isSpaceChar(ch))
			return false;
		return true;
	}

	public void tokenize(Sentence sentence) {
		String text = sentence.getText();
		int start = 0;
		for (int i = 1; i - 1 < text.length(); i++) {
			char current = text.charAt(i - 1);
			char next = 0;
			if (i < text.length())
				next = text.charAt(i);
			// System.out.println("Current =\"" + current + "\" and next = \"" + next + "\"");
			// System.out.println("Current =\"" + Character.isLetter(current) + "\" and next = \"" + Character.isLetter(next) + "\"");
			if (Character.isSpaceChar(current)) {
				start = i;
			} else if (Character.isLetter(current)) {
				if (!Character.isLetter(next) || (Character.isLowerCase(current) && Character.isUpperCase(next))) {
					sentence.addToken(new Token(sentence, start, i));
					start = i;
				}
			} else if (Character.isDigit(current)) {
				if (!Character.isDigit(next)) {
					sentence.addToken(new Token(sentence, start, i));
					start = i;
				}
			} else if (isPunctuation(current)) {
				sentence.addToken(new Token(sentence, start, i));
				start = i;
			}
		}
		if (start < text.length())
			sentence.addToken(new Token(sentence, start, text.length()));
	}

	public List<String> getTokens(String text) {
		List<String> tokens = new ArrayList<String>();
		int start = 0;
		for (int i = 1; i - 1 < text.length(); i++) {
			char current = text.charAt(i - 1);
			char next = 0;
			if (i < text.length())
				next = text.charAt(i);
			if (Character.isSpaceChar(current)) {
				start = i;
			} else if (Character.isLetter(current)) {
				if (!Character.isLetter(next)) {
					tokens.add(text.substring(start, i));
					start = i;
				}
			} else if (Character.isDigit(current)) {
				if (!Character.isDigit(next)) {
					tokens.add(text.substring(start, i));
					start = i;
				}
			} else if (isPunctuation(current)) {
				tokens.add(text.substring(start, i));
				start = i;
			}
		}
		if (start < text.length())
			tokens.add(text.substring(start, text.length()));
		return tokens;
	}
}