package de.darkandblue.neuralnetwork.initialization;

public abstract class Distribution {
  public abstract double biasFrom();
  
  public abstract double biasTo();
  
  public abstract double weightsFrom();
  
  public abstract double weightsTo();
}