package com.ibm.research.ai.irsimple;

import java.nio.file.*;

import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.standard.*;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;

public class Search {
    final Directory directory;
    final DirectoryReader ireader;
    final IndexSearcher isearcher;
    final QueryParser parser;
    PollingThreadedExecutor threads;
    final int topN;
    final float minScore;
    final float maxScore;
    final double targetNoAnswerFraction;
    
    //CONSIDER: AtomicInteger?
    double noAnswerCount;
    double hasAnswerCount;
       
    public Search(String indexDir, int topN, float minScore, float maxScore, float targetNoAnswerFraction) {
        try {
            Analyzer analyzer = new StandardAnalyzer();
            //directory = FSDirectory.open(Paths.get(opts.indexDir)); 
            directory = new MMapDirectory(Paths.get(indexDir));
            ireader = DirectoryReader.open(directory);
            isearcher = new IndexSearcher(ireader);
            // Parse a simple query that searches "text":
            parser = new QueryParser("text", analyzer);
        } catch (Exception e) {
            throw new Error(e);
        }
        this.topN = topN;
        this.minScore = minScore;
        this.maxScore = maxScore;
        
        this.noAnswerCount = 0;
        this.hasAnswerCount = 0;
        this.targetNoAnswerFraction = targetNoAnswerFraction;
    }
    
    static final String specialChars = "\\+-!():^[]\"{}~*?|&/";
    static String escapeQuery(String query) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < query.length(); i++) {
            char c = query.charAt(i);
            if (specialChars.indexOf(c) != -1)
                sb.append('\\');
            sb.append(c);
        }
        return (' '+sb.toString()+' ').replaceAll("\\s([Nn][Oo][Tt]|[Aa][Nn][Dd]|[Oo][Rr])\\s", " ").trim();
    }
    
    static String normalize(String txt) {
        return " "+txt.trim().toLowerCase().replaceAll("[\\W\\s]+", " ")+" ";
    }
    
    private boolean allowNoAnswer() {
        if (this.hasAnswerCount < 100)
            return false;
        double currentNoAnswerFraction = this.noAnswerCount / (this.noAnswerCount + this.hasAnswerCount);
        return currentNoAnswerFraction < targetNoAnswerFraction && 
               Math.random() < 3*(targetNoAnswerFraction - currentNoAnswerFraction);
    }

    public String query(String q, String excludeId, String answer) {
        String mustContainNormalized = normalize(answer);
        try {
            Query query = parser.parse(escapeQuery(q));
            ScoreDoc[] hits = isearcher.search(query, topN).scoreDocs;

            for (ScoreDoc hit : hits) {
                org.apache.lucene.document.Document hitDoc = isearcher.doc(hit.doc);
                if (hit.score < minScore || hit.score > maxScore)
                    continue;
                if (hitDoc.get("id").equals(excludeId))
                    continue;
                String text = hitDoc.get("text");
                
                if (normalize(text).contains(mustContainNormalized)) {
                    this.hasAnswerCount += 1;
                    return text;
                } else if (allowNoAnswer()) {
                    this.noAnswerCount += 1;
                    return text;
                }
            }
        } catch (Exception e) {
            //ignore
        }
        return null;
    }
}
