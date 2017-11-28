package dictionary;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import dictionary.Mention.MentionType;;


public class DictionaryTagger {
	
	
	private boolean filterContainedMentions;
	protected Trie<String, HashSet<EntityType>> entities;
	protected Trie<String, Boolean> notInclude;
	private boolean normalizeMixedCase;
	private boolean normalizeDigits;
	private boolean generate2PartVariations;
	private boolean dropEndParentheticals;
	private FineTokenizer tokenizer;
	private EntityType dictionaryType;
	
	/**
	 * Creates a new {@link DictionaryTagger}
	 */
	public DictionaryTagger() {
		entities = new Trie<String, HashSet<EntityType>>();
		notInclude = new Trie<String, Boolean>();
		tokenizer = new FineTokenizer();
	}
	
	public void configure(){
		System.out.println("Configuring the Dictionary Tagger ...");
		
		filterContainedMentions = true;
		normalizeDigits = false;
		normalizeMixedCase = true;
		generate2PartVariations = false;
		dropEndParentheticals = true;
		dictionaryType = EntityType.getType("drug");
	}
	
	public void load(String filename) throws IOException{
		
		System.out.println("Loading the dictionary from file: " + filename + " ...");
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		
		String line = br.readLine();
		
		while(line != null){
			line = line.trim();
			if (line.length() > 0) {
				add(line, dictionaryType);
			}
			line = br.readLine();
		}
		br.close();
	}
	
	public void add(String text, EntityType type) {
		add(text, Collections.singleton(type));
	}
	
	public void add(String text, Collection<EntityType> types) {
		// TODO Make configurable
		// if (text.length() == 1)
		// return;
		// TODO Add ability to not add items over N (eg 10) tokens long
		List<String> tokens = process(text);
		add(tokens, types);
		if (generate2PartVariations) {
			if (tokens.size() == 1 && tokens.get(0).matches("[A-Za-z]+[0-9]+")) {
				int split = 0;
				String token = tokens.get(0);
				while (Character.isLetter(token.charAt(split)))
					split++;
				add2Part(token.substring(0, split), token.substring(split, token.length()), types);
			}
			if (tokens.size() == 2) {
				add2Part(tokens.get(0), tokens.get(1), types);
			}
			if (tokens.size() == 3 && (tokens.get(1).equals("-") || tokens.get(1).equals("/"))) {
				add2Part(tokens.get(0), tokens.get(2), types);
			}
		}
		// TODO These lines add GENE recall but drop precision
		// if (tokens.size() > 1 && tokens.get(tokens.size() -
		// 1).equals("homolog"))
		// add(tokens.subList(0, tokens.size() - 1), types);
	}
	
	public boolean add(List<String> tokens, Collection<EntityType> types) {
		if (tokens.size() == 0)
			throw new IllegalArgumentException("Number of tokens must be greater than zero");
		// Verify that the sequence to be added is not listed as not included
		Boolean value = notInclude.getValue(tokens);
		if (value != null)
			return false;
		// If configured, drop parenthetical phrases at the end of the sequence
		if (dropEndParentheticals && tokens.get(tokens.size() - 1).equals(")")) {
			int openParen = tokens.size() - 1;
			while (openParen > 0 && !tokens.get(openParen).equals("("))
				openParen--;
			if (openParen <= 0)
				return false;
			tokens = tokens.subList(0, openParen);
		}
		HashSet<EntityType> currentTypes = entities.getValue(tokens);
		if (currentTypes == null) {
			currentTypes = new HashSet<EntityType>(1);
			entities.add(tokens, currentTypes);
		}
		return currentTypes.addAll(types);
	}
	
	
	private void add2Part(String part1, String part2, Collection<EntityType> types) {
		List<String> tokens = new ArrayList<String>();
		tokens.add(part1 + part2);
		tokens.add(part2);
		add(tokens, types);
		tokens = new ArrayList<String>();
		tokens.add(part1);
		tokens.add(part2);
		add(tokens, types);
		tokens.add(1, "-");
		add(tokens, types);
		tokens.set(1, "/");
		add(tokens, types);
	}
	
	
	
	protected List<String> process(String input) {
		if (input == null)
			throw new IllegalArgumentException();
		List<String> tokens = tokenizer.getTokens(input);
		for (int i = 0; i < tokens.size(); i++)
			tokens.set(i, transform(tokens.get(i)));
		return tokens;
	}
	
	protected String transform(String str) {
		// This has been optimized for very fast operation
		String result = str;

		if (normalizeMixedCase || normalizeDigits) {
			char[] chars = str.toCharArray();
			if (normalizeMixedCase) {
				boolean hasUpper = false;
				boolean hasLower = false;
				for (int i = 0; i < chars.length && (!hasUpper || !hasLower); i++) {
					hasUpper |= Character.isUpperCase(chars[i]);
					hasLower |= Character.isLowerCase(chars[i]);
				}
				if (hasUpper && hasLower)
					for (int i = 0; i < chars.length; i++)
						chars[i] = Character.toLowerCase(chars[i]);
			}
			// Note that this only works on single digits
			if (normalizeDigits)
				for (int i = 0; i < chars.length; i++)
					if (Character.isDigit(chars[i]))
						chars[i] = '0';
			result = new String(chars);
		}
		return result;
	}
	
	public void tag(Sentence sentence) {
		List<Token> tokens = sentence.getTokens();
		// Lookup mentions
		List<Mention> mentions = new LinkedList<Mention>();
		for (int startIndex = 0; startIndex < tokens.size(); startIndex++) {
			Trie<String, HashSet<EntityType>> t = entities;
			for (int currentIndex = startIndex; currentIndex < tokens.size() && t != null; currentIndex++) {
				HashSet<EntityType> entityTypes = t.getValue();
				if (entityTypes != null)
					for (EntityType entityType : entityTypes)
						mentions.add(new Mention(sentence, startIndex, currentIndex, entityType, MentionType.Found));
				Token currentToken = tokens.get(currentIndex);
				t = t.getChild(transform(currentToken.getText()));
			}
		}

		// Add mentions found

		// Iterator<Mention> mentionIterator = mentions.iterator();
		// while (mentionIterator.hasNext())
		// {
		// Mention mention = mentionIterator.next();
		// boolean contained = false;
		// for (Mention mention2 : mentions)
		// contained |= !mention2.equals(mention) && mention2.contains(mention);
		// if (!filterContainedMentions || !contained)
		// sentence.addMention(mention);
		// }

		if (filterContainedMentions) {
			while (!mentions.isEmpty()) {
				Mention mention1 = mentions.remove(0);
				int start = mention1.getStart();
				int end = mention1.getEnd();
				ArrayList<Mention> adjacentMentions = new ArrayList<Mention>();
				Iterator<Mention> mentionIterator = mentions.iterator();
				boolean changed = true;
				while (changed) {
					changed = false;
					while (mentionIterator.hasNext()) {
						Mention mention2 = mentionIterator.next();
						boolean adjacent = (end >= mention2.getStart()) && (start <= mention2.getEnd());
						if (mention1.getEntityType().equals(mention2.getEntityType()) && adjacent) {
							adjacentMentions.add(mention2);
							mentionIterator.remove();
							start = Math.min(start, mention2.getStart());
							end = Math.max(end, mention2.getEnd());
							changed = true;
						}
					}
				}
				sentence.addMention(new Mention(sentence, start, end, mention1.getEntityType(), MentionType.Found));
			}
		} else {
			for (Mention mention : mentions)
				sentence.addMention(mention);
		}

		// System.out.println(sentence.getText());
		// for (Mention mention : sentence.getMentions())
		// System.out.println("\t" + mention.getText());
	}

	public void suppress(String text) {
		notInclude.add(process(text), Boolean.TRUE);
	}

	/**
	 * @return The number of entries in this dictionary
	 */
	public int size() {
		// TODO PERFORMANCE This is a very intensive operation due to having to
		// search the entire tree!
		return entities.size();
	}

	public boolean isFilterContainedMentions() {
		return filterContainedMentions;
	}

	public Trie<String, HashSet<EntityType>> getEntities() {
		return entities;
	}

	public Trie<String, Boolean> getNotInclude() {
		return notInclude;
	}

	public boolean isNormalizeMixedCase() {
		return normalizeMixedCase;
	}

	public boolean isNormalizeDigits() {
		return normalizeDigits;
	}

	public boolean isGenerate2PartVariations() {
		return generate2PartVariations;
	}

	public boolean isDropEndParentheticals() {
		return dropEndParentheticals;
	}

	public FineTokenizer getTokenizer() {
		return tokenizer;
	}

	public EntityType getDictionaryType() {
		return dictionaryType;
	}

	public void setFilterContainedMentions(boolean filterContainedMentions) {
		this.filterContainedMentions = filterContainedMentions;
	}

	public void setEntities(Trie<String, HashSet<EntityType>> entities) {
		this.entities = entities;
	}

	public void setNotInclude(Trie<String, Boolean> notInclude) {
		this.notInclude = notInclude;
	}

	public void setNormalizeMixedCase(boolean normalizeMixedCase) {
		this.normalizeMixedCase = normalizeMixedCase;
	}

	public void setNormalizeDigits(boolean normalizeDigits) {
		this.normalizeDigits = normalizeDigits;
	}

	public void setGenerate2PartVariations(boolean generate2PartVariations) {
		this.generate2PartVariations = generate2PartVariations;
	}

	public void setDropEndParentheticals(boolean dropEndParentheticals) {
		this.dropEndParentheticals = dropEndParentheticals;
	}

	public void setTokenizer(FineTokenizer tokenizer) {
		this.tokenizer = tokenizer;
	}

	public void setDictionaryType(EntityType dictionaryType) {
		this.dictionaryType = dictionaryType;
	}
	
	
	
}
