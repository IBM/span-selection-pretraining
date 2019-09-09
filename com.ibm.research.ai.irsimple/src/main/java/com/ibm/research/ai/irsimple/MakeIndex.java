package com.ibm.research.ai.irsimple;

import java.io.*;
import java.nio.file.*;
import java.text.*;

import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.standard.*;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;

import com.google.gson.Gson;

public class MakeIndex {
    
    public static class JsonDocument {
        /**
         * Same structure as Anserini document
         */
        public String id;
        public String contents;
    }
    
    public static void main(String[] args) throws Exception {
        String docDir = args[0];
        String indexDir = args[1];
        //build index
        Analyzer analyzer = new StandardAnalyzer();
        // Store an index on disk     
        Directory directory = FSDirectory.open(Paths.get(indexDir));
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter iwriter = new IndexWriter(directory, config);
        PollingThreadedExecutor threads = new PollingThreadedExecutor();
        Gson gson = new Gson();
        for (File f : new FileUtil.FileIterable(new File(docDir))) {
            for (String line : new FileUtil.FileLineIterable(f, true)) {
                threads.execute(() -> {
                    JsonDocument doc = gson.fromJson(line, JsonDocument.class);
                    //remove unicode accents
                    doc.contents = Normalizer.normalize(doc.contents, Normalizer.Form.NFD).replaceAll("\\p{M}", "");

                    Document ldoc = new Document();
                    ldoc.add(new Field("text", doc.contents, TextField.TYPE_STORED));
                    ldoc.add(new StoredField("id", doc.id));
                    try {
                        iwriter.addDocument(ldoc);
                    } catch (IOException ioe) {
                        System.err.println(ioe.getMessage());
                    }  
                });
            }
        }
        threads.awaitFinishing();
        iwriter.close();
    }
}
