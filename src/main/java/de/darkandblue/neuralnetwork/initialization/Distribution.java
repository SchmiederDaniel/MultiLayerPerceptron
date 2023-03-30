package de.darkandblue.neuralnetwork.initialization;

public abstract class Distribution {
  public abstract float biasFrom();
  
  public abstract float biasTo();
  
  public abstract float weightsFrom();
  
  public abstract float weightsTo();
}