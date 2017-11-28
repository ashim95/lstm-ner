package dictionary;

public class SentenceWithOffset extends Sentence {

	private int offset;

	public SentenceWithOffset(String sentenceId, String documentId, String text, int offset) {
		super(sentenceId, documentId, text);
		this.offset = offset;
	}

	public int getOffset() {
		return offset;
	}

	@Override
	public Sentence copy(boolean includeTokens, boolean includeMentions) {
		Sentence sentence2 = new SentenceWithOffset(getSentenceId(), getDocumentId(), getText(), offset);
		if (includeTokens) {
			for (Token token : getTokens())
				sentence2.addToken(new Token(sentence2, token.getStart(), token.getEnd()));
		}
		if (includeMentions) {
			for (Mention mention : getMentions())
				sentence2.addMention(mention.copy(sentence2));
		}
		return sentence2;
	}
}
