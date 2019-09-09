package com.ibm.research.ai.irsimple;

import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class PollingThreadedExecutor {
	public ThreadPoolExecutor threadPool;
	private int numProcs;
	private int queueDepth;
	private String name;
	
	public int pollDelayMilli = 100;
	
	protected AtomicBoolean hasError = new AtomicBoolean(false);
	protected Throwable error;
	
	public int getNumProcessors() {
		return numProcs;
	}
	
	class UncaughtExceptionHandler implements Thread.UncaughtExceptionHandler {
        @Override
        public void uncaughtException(Thread t, Throwable e) {
            if (!hasError.getAndSet(true))
                error = e;
        }
	}
	
	private void buildThreadPool() {
		BlockingQueue<Runnable> q = queueDepth == Integer.MAX_VALUE ? 
				new LinkedBlockingQueue<>() : 
				new ArrayBlockingQueue<Runnable>(queueDepth*numProcs);
		threadPool = new ThreadPoolExecutor(numProcs, numProcs, 10, TimeUnit.MINUTES, q,
				new ThreadFactory() {
		            int threadCount = 0;
					@Override
					public Thread newThread(Runnable runnable) {
						Thread thread = Executors.defaultThreadFactory().newThread(runnable);
						thread.setDaemon(true);
						thread.setUncaughtExceptionHandler(new UncaughtExceptionHandler());
						if (name != null)
						    thread.setName(name+"-"+threadCount);
						++threadCount;
						return thread;
					}
				}, new ThreadPoolExecutor.AbortPolicy());
	}
	
	public PollingThreadedExecutor() {
		this.queueDepth = 10;
		numProcs = Runtime.getRuntime().availableProcessors();
		buildThreadPool();
	}
	
	public PollingThreadedExecutor(String name, int queueDepth, int numThreads) {
		this.queueDepth = queueDepth;
		numProcs = numThreads;
		this.name = name;
		buildThreadPool();
	}

	public boolean isFinished() {
		return (threadPool.getQueue().isEmpty() && threadPool.getActiveCount() == 0);
	}
	
	public void awaitFinishing(long milliPoll) {
		try {
			while (!isFinished()) 
				Thread.sleep(milliPoll);	
		} catch (Exception e) { throw new Error(e);}
	}
	
	public void awaitFinishing() {
		awaitFinishing(pollDelayMilli);
	}
	
	public void execute(Runnable task) {
		while (true) {
		    if (this.hasError.get() && error != null) {
		        if (error instanceof RuntimeException)
		            throw (RuntimeException)error;
		        else if (error instanceof Error)
		            throw (Error)error;
		        throw new RuntimeException(error);
		    }
			try {
				threadPool.execute(task);
				break;
			} catch (RejectedExecutionException e) {
				try {
					Thread.sleep(pollDelayMilli);
				} catch (InterruptedException e1) {
					throw new Error(e1);
				}
			}
		}
	}
	
	public void shutdown() {	
		threadPool.shutdown();
		try {while (!threadPool.awaitTermination(100, TimeUnit.DAYS)); } catch (Exception e) {throw new Error(e);}
	}

}
