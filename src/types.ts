export type ActivationFunction = 'relu' | 'sigmoid';

export interface TokenData {
  id: string;
  text: string;
  embedding: number[];
  position: { x: number; y: number };
}

export interface Node {
  id: string;
  layer: number;
  value: number;
  gradient?: number;
}

export interface Link {
  source: string;
  target: string;
  weight: number;
  gradient?: number;
}
