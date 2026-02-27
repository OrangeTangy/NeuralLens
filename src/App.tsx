import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import * as d3 from 'd3';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Html } from '@react-three/drei';
import * as THREE from 'three';
import { cn } from './lib/utils';
import { TokenData, ActivationFunction, Node, Link } from './types';
import { GoogleGenAI } from "@google/genai";
import Markdown from 'react-markdown';
import { 
  Cpu, 
  Layers, 
  Network, 
  ArrowRight, 
  Play, 
  RotateCcw, 
  Settings2,
  Info,
  ChevronRight,
  Zap,
  MessageSquare,
  Send,
  Loader2,
  Sparkles
} from 'lucide-react';

// --- Mock Data & Helpers ---
const generateEmbedding = (text: string) => {
  const seed = text.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const random = d3.randomNormal.source(d3.randomLcg(seed / 1000))(0, 1);
  return Array.from({ length: 8 }, () => random());
};

const relu = (x: number) => Math.max(0, x);
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

// --- Components ---

const Tokenizer = ({ text, onTokensChange }: { text: string, onTokensChange: (tokens: TokenData[]) => void }) => {
  const [tokens, setTokens] = useState<TokenData[]>([]);

  useEffect(() => {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const newTokens = words.map((word, i) => ({
      id: `${word}-${i}`,
      text: word,
      embedding: generateEmbedding(word),
      position: { x: Math.random() * 100, y: Math.random() * 100 }
    }));
    setTokens(newTokens);
    onTokensChange(newTokens);
  }, [text]);

  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-4 h-4 text-neural-accent" />
        <span className="mono-label">Step 01: Tokenization</span>
      </div>
      <div className="flex-1 flex flex-wrap gap-2 content-start overflow-y-auto custom-scrollbar">
        <AnimatePresence mode="popLayout">
          {tokens.map((token, i) => (
            <motion.div
              key={token.id}
              initial={{ opacity: 0, scale: 0.8, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ delay: i * 0.05 }}
              className="px-3 py-1.5 bg-neural-accent/10 border border-neural-accent/30 rounded text-sm font-mono text-neural-accent"
            >
              {token.text}
            </motion.div>
          ))}
        </AnimatePresence>
        {text.length === 0 && (
          <div className="text-neural-muted text-sm italic">Start typing to see tokens...</div>
        )}
      </div>
    </div>
  );
};

const TokenPoint = ({ token }: { token: TokenData }) => {
  // Use first 3 dimensions for 3D space
  const x = (token.embedding[0] || 0) * 3;
  const y = (token.embedding[1] || 0) * 3;
  const z = (token.embedding[2] || 0) * 3;
  const position = new THREE.Vector3(x, y, z);
  const origin = new THREE.Vector3(0, 0, 0);
  const length = position.length();
  const direction = position.clone().normalize();

  return (
    <group>
      {/* Arrow from origin to point */}
      <primitive 
        object={new THREE.ArrowHelper(direction, origin, length, '#3b82f6', 0.2, 0.1)} 
      />
      
      {/* The Point */}
      <mesh position={[x, y, z]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshBasicMaterial color="#3b82f6" />
      </mesh>

      {/* HTML Label - Robust and doesn't need external fonts */}
      <Html position={[x, y + 0.2, z]} center distanceFactor={10}>
        <div className="px-1.5 py-0.5 bg-neural-panel/90 border border-neural-accent/30 rounded text-[10px] font-mono text-neural-accent whitespace-nowrap pointer-events-none shadow-lg">
          {token.text}
        </div>
      </Html>
    </group>
  );
};

const VectorSpace = ({ tokens }: { tokens: TokenData[] }) => {
  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Layers className="w-4 h-4 text-neural-accent" />
        <span className="mono-label">Step 02: Embedding Space (3D Projection)</span>
      </div>
      <div className="flex-1 relative bg-black/40 rounded-lg overflow-hidden border border-neural-border">
        <React.Suspense fallback={<div className="w-full h-full flex items-center justify-center text-neural-muted text-xs">Loading 3D Engine...</div>}>
          <Canvas 
            camera={{ position: [8, 8, 8], fov: 45 }}
            dpr={[1, 2]}
          >
            <color attach="background" args={['#0a0a0b']} />
            <ambientLight intensity={1} />
            
            <group>
              {tokens.map((token) => (
                <TokenPoint key={token.id} token={token} />
              ))}
            </group>

            <gridHelper args={[20, 20, '#27272a', '#161618']} />
            <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
          </Canvas>
        </React.Suspense>
        
        <div className="absolute bottom-3 right-3 flex flex-col items-end gap-1 pointer-events-none z-10">
          <div className="text-[8px] font-mono text-neural-muted uppercase">Drag to Rotate</div>
          <div className="text-[8px] font-mono text-neural-muted uppercase">Scroll to Zoom</div>
        </div>
      </div>
    </div>
  );
};

const NeuralNetworkVisualizer = ({ 
  activation, 
  isForwarding, 
  isBackpropping 
}: { 
  activation: ActivationFunction,
  isForwarding: boolean,
  isBackpropping: boolean
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [links, setLinks] = useState<Link[]>([]);

  useEffect(() => {
    const layerSizes = [4, 6, 6, 2];
    const newNodes: Node[] = [];
    const newLinks: Link[] = [];

    layerSizes.forEach((size, layerIdx) => {
      for (let i = 0; i < size; i++) {
        newNodes.push({
          id: `node-${layerIdx}-${i}`,
          layer: layerIdx,
          value: Math.random(),
          gradient: 0
        });
      }
    });

    for (let l = 0; l < layerSizes.length - 1; l++) {
      for (let i = 0; i < layerSizes[l]; i++) {
        for (let j = 0; j < layerSizes[l + 1]; j++) {
          newLinks.push({
            source: `node-${l}-${i}`,
            target: `node-${l + 1}-${j}`,
            weight: (Math.random() - 0.5) * 2,
            gradient: 0
          });
        }
      }
    }

    setNodes(newNodes);
    setLinks(newLinks);
  }, []);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    svg.selectAll("*").remove();

    const layerCount = 4;
    const xPadding = 80;
    const yPadding = 40;
    const xScale = d3.scaleLinear().domain([0, layerCount - 1]).range([xPadding, width - xPadding]);

    const getNodesInLayer = (l: number) => nodes.filter(n => n.layer === l);

    // Draw Links
    const linkGroup = svg.append("g").attr("class", "links");
    linkGroup.selectAll<SVGLineElement, Link>("line")
      .data(links)
      .join("line")
      .attr("class", "network-link")
      .attr("x1", (d: Link) => {
        const source = nodes.find(n => n.id === d.source)!;
        return xScale(source.layer);
      })
      .attr("y1", (d: Link) => {
        const source = nodes.find(n => n.id === d.source)!;
        const layerNodes = getNodesInLayer(source.layer);
        const yScale = d3.scaleLinear().domain([0, layerNodes.length - 1]).range([yPadding, height - yPadding]);
        return yScale(layerNodes.indexOf(source));
      })
      .attr("x2", (d: Link) => {
        const target = nodes.find(n => n.id === d.target)!;
        return xScale(target.layer);
      })
      .attr("y2", (d: Link) => {
        const target = nodes.find(n => n.id === d.target)!;
        const layerNodes = getNodesInLayer(target.layer);
        const yScale = d3.scaleLinear().domain([0, layerNodes.length - 1]).range([yPadding, height - yPadding]);
        return yScale(layerNodes.indexOf(target));
      })
      .attr("stroke", (d: Link) => d.weight > 0 ? "#3b82f6" : "#ef4444")
      .attr("stroke-width", (d: Link) => Math.abs(d.weight) * 2)
      .attr("stroke-opacity", isForwarding || isBackpropping ? 0.4 : 0.1)
      .style("filter", isForwarding || isBackpropping ? "drop-shadow(0 0 2px currentColor)" : "none");

    // Draw Nodes
    const nodeGroup = svg.append("g").attr("class", "nodes");
    nodes.forEach(node => {
      const layerNodes = getNodesInLayer(node.layer);
      const yScale = d3.scaleLinear().domain([0, layerNodes.length - 1]).range([yPadding, height - yPadding]);
      const x = xScale(node.layer);
      const y = yScale(layerNodes.indexOf(node));

      const g = nodeGroup.append("g").attr("transform", `translate(${x}, ${y})`);
      
      g.append("circle")
        .attr("r", 8)
        .attr("fill", "#161618")
        .attr("stroke", isForwarding || isBackpropping ? "#3b82f6" : "#27272a")
        .attr("stroke-width", 2)
        .style("filter", isForwarding || isBackpropping ? "drop-shadow(0 0 4px #3b82f6)" : "none");

      g.append("circle")
        .attr("r", 6)
        .attr("fill", activation === 'relu' ? (node.value > 0 ? "#3b82f6" : "#161618") : d3.interpolateRdBu(node.value))
        .attr("opacity", 0.8);
    });

    // Animations
    if (isForwarding) {
      const particles = 30;
      for (let i = 0; i < particles; i++) {
        const link = links[Math.floor(Math.random() * links.length)];
        const source = nodes.find(n => n.id === link.source)!;
        const target = nodes.find(n => n.id === link.target)!;
        
        const sLayerNodes = getNodesInLayer(source.layer);
        const tLayerNodes = getNodesInLayer(target.layer);
        const syScale = d3.scaleLinear().domain([0, sLayerNodes.length - 1]).range([yPadding, height - yPadding]);
        const tyScale = d3.scaleLinear().domain([0, tLayerNodes.length - 1]).range([yPadding, height - yPadding]);

        const sx = xScale(source.layer);
        const sy = syScale(sLayerNodes.indexOf(source));
        const tx = xScale(target.layer);
        const ty = tyScale(tLayerNodes.indexOf(target));

        svg.append("circle")
          .attr("r", 3)
          .attr("fill", "#3b82f6")
          .attr("cx", sx)
          .attr("cy", sy)
          .style("filter", "drop-shadow(0 0 5px #3b82f6)")
          .transition()
          .duration(400 + Math.random() * 400)
          .attr("cx", tx)
          .attr("cy", ty)
          .remove();
      }
    }

    if (isBackpropping) {
       const particles = 30;
      for (let i = 0; i < particles; i++) {
        const link = links[Math.floor(Math.random() * links.length)];
        const source = nodes.find(n => n.id === link.source)!;
        const target = nodes.find(n => n.id === link.target)!;
        
        const sLayerNodes = getNodesInLayer(source.layer);
        const tLayerNodes = getNodesInLayer(target.layer);
        const syScale = d3.scaleLinear().domain([0, sLayerNodes.length - 1]).range([yPadding, height - yPadding]);
        const tyScale = d3.scaleLinear().domain([0, tLayerNodes.length - 1]).range([yPadding, height - yPadding]);

        const sx = xScale(source.layer);
        const sy = syScale(sLayerNodes.indexOf(source));
        const tx = xScale(target.layer);
        const ty = tyScale(tLayerNodes.indexOf(target));

        svg.append("circle")
          .attr("r", 3)
          .attr("fill", "#ef4444")
          .attr("cx", tx)
          .attr("cy", ty)
          .style("filter", "drop-shadow(0 0 5px #ef4444)")
          .transition()
          .duration(400 + Math.random() * 400)
          .attr("cx", sx)
          .attr("cy", sy)
          .remove();
      }
    }

  }, [nodes, links, activation, isForwarding, isBackpropping]);

  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-neural-accent" />
          <span className="mono-label">Step 03: Forward & Backprop</span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div className={cn("w-2 h-2 rounded-full", isForwarding ? "bg-neural-accent animate-pulse" : "bg-neural-border")} />
            <span className="text-[10px] text-neural-muted uppercase">Forward</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className={cn("w-2 h-2 rounded-full", isBackpropping ? "bg-neural-error animate-pulse" : "bg-neural-border")} />
            <span className="text-[10px] text-neural-muted uppercase">Backprop</span>
          </div>
        </div>
      </div>
      <div className="flex-1 relative">
        <svg ref={svgRef} className="w-full h-full" />
      </div>
    </div>
  );
};

const MathPanel = ({ activation, tokens }: { activation: ActivationFunction, tokens: TokenData[] }) => {
  const firstToken = tokens[0];
  
  return (
    <div className="glass-panel p-6 h-full flex flex-col gap-4 overflow-y-auto custom-scrollbar">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-4 h-4 text-neural-accent flex items-center justify-center font-serif italic font-bold">Σ</div>
        <span className="mono-label">Related Math</span>
      </div>
      
      <div className="space-y-6">
        {firstToken && (
          <section className="space-y-2">
            <h4 className="text-xs font-semibold text-neural-warning uppercase tracking-wider">Concrete Trace: "{firstToken.text}"</h4>
            <div className="bg-neural-accent/5 p-3 rounded border border-neural-accent/20 font-mono text-[10px] leading-relaxed">
              <p className="text-neural-muted mb-1">1. Token ID: <span className="text-neural-accent">{firstToken.id.split('-')[1]}</span></p>
              <p className="text-neural-muted mb-1">2. Embedding (first 3 dims):</p>
              <div className="flex gap-2 text-neural-accent mb-2">
                [{firstToken.embedding.slice(0, 3).map(v => v.toFixed(3)).join(', ')}...]
              </div>
              <p className="text-neural-muted mb-1">3. Forward Pass (Layer 1):</p>
              <p className="text-neural-muted italic">z = Σ(w &middot; a) + b</p>
              <p className="text-neural-muted italic">a = {activation}(z)</p>
            </div>
          </section>
        )}

        <section className="space-y-2">
          <h4 className="text-xs font-semibold text-neural-accent uppercase tracking-wider">1. Embedding Lookup</h4>
          <div className="bg-black/40 p-3 rounded border border-neural-border font-mono text-xs leading-relaxed">
            <p className="text-neural-muted mb-2">// Vector representation</p>
            <div className="text-center py-2 text-sm">
              v = W<sub>e</sub> &middot; x
            </div>
            <p className="text-[10px] text-neural-muted mt-2 italic">
              Where x is a one-hot encoded token and W<sub>e</sub> is the embedding weight matrix.
            </p>
          </div>
        </section>

        <section className="space-y-2">
          <h4 className="text-xs font-semibold text-neural-accent uppercase tracking-wider">2. Activation: {activation.toUpperCase()}</h4>
          <div className="bg-black/40 p-3 rounded border border-neural-border font-mono text-xs leading-relaxed">
            {activation === 'relu' ? (
              <>
                <div className="text-center py-2 text-sm">
                  f(z) = max(0, z)
                </div>
                <div className="text-center py-1 text-[10px] text-neural-muted border-t border-neural-border/30 mt-2">
                  Derivative: f'(z) = 1 if z &gt; 0 else 0
                </div>
              </>
            ) : (
              <>
                <div className="text-center py-2 text-sm">
                  f(z) = 1 / (1 + e<sup>-z</sup>)
                </div>
                <div className="text-center py-1 text-[10px] text-neural-muted border-t border-neural-border/30 mt-2">
                  Derivative: f'(z) = f(z)(1 - f(z))
                </div>
              </>
            )}
            <p className="text-[10px] text-neural-muted mt-2 italic">
              Non-linearity allows the network to approximate complex functions.
            </p>
          </div>
        </section>

        <section className="space-y-2">
          <h4 className="text-xs font-semibold text-neural-accent uppercase tracking-wider">3. Softmax (Output Layer)</h4>
          <div className="bg-black/40 p-3 rounded border border-neural-border font-mono text-xs leading-relaxed">
            <div className="text-center py-2 text-sm">
              σ(z)<sub>i</sub> = e<sup>z<sub>i</sub></sup> / Σ e<sup>z<sub>j</sub></sup>
            </div>
            <p className="text-[10px] text-neural-muted mt-2 italic">
              Converts raw scores (logits) into a probability distribution.
            </p>
          </div>
        </section>

        <section className="space-y-2">
          <h4 className="text-xs font-semibold text-neural-error uppercase tracking-wider">4. Cross Entropy Loss</h4>
          <div className="bg-black/40 p-3 rounded border border-neural-border font-mono text-xs leading-relaxed">
            <div className="text-center py-2 text-sm">
              L = -Σ y<sub>i</sub> log(ŷ<sub>i</sub>)
            </div>
            <p className="text-[10px] text-neural-muted mt-2 italic">
              Measures the distance between predicted probability ŷ and true label y.
            </p>
          </div>
        </section>

        <section className="space-y-2">
          <h4 className="text-xs font-semibold text-neural-success uppercase tracking-wider">5. Gradient Descent</h4>
          <div className="bg-black/40 p-3 rounded border border-neural-border font-mono text-xs leading-relaxed">
            <div className="text-center py-2 text-sm">
              w<sub>t+1</sub> = w<sub>t</sub> - η &nabla;<sub>w</sub>L
            </div>
            <p className="text-[10px] text-neural-muted mt-2 italic">
              Weights are updated by moving in the opposite direction of the gradient &nabla;L with learning rate η.
            </p>
          </div>
        </section>

        <section className="space-y-2">
          <h4 className="text-xs font-semibold text-neural-warning uppercase tracking-wider">6. The Chain Rule</h4>
          <div className="bg-black/40 p-3 rounded border border-neural-border font-mono text-xs leading-relaxed">
            <div className="text-center py-2 text-sm">
              ∂L/∂w = (∂L/∂a) &middot; (∂a/∂z) &middot; (∂z/∂w)
            </div>
            <p className="text-[10px] text-neural-muted mt-2 italic">
              The engine of backpropagation: decomposing the gradient into local derivatives.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
};

const AIChat = ({ onAction }: { onAction: (action: string) => void }) => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleAsk = async () => {
    if (!query.trim() || isLoading) return;
    setIsLoading(true);
    setResponse("");

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const model = "gemini-3-flash-preview";
      
      const prompt = `You are an expert in Neural Networks and LLMs. 
      The user is interacting with a visualizer called "Neural Lens".
      Answer the user's question concisely. 
      If you want to trigger a visual action, include one of these tags at the end of your response: 
      [ACTION:FORWARD], [ACTION:BACKPROP], [ACTION:RELU], [ACTION:SIGMOID].
      
      User Question: ${query}`;

      const result = await ai.models.generateContent({
        model,
        contents: [{ parts: [{ text: prompt }] }],
      });

      const text = result.text || "I'm sorry, I couldn't process that.";
      setResponse(text);

      // Check for actions
      if (text.includes("[ACTION:FORWARD]")) onAction("forward");
      if (text.includes("[ACTION:BACKPROP]")) onAction("backprop");
      if (text.includes("[ACTION:RELU]")) onAction("relu");
      if (text.includes("[ACTION:SIGMOID]")) onAction("sigmoid");

    } catch (error) {
      console.error("AI Error:", error);
      setResponse("Error connecting to the neural oracle. Please check your connection.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="glass-panel p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-neural-accent" />
          <span className="mono-label">Neural Oracle (AI Chat)</span>
        </div>
        {isLoading && <Loader2 className="w-4 h-4 text-neural-accent animate-spin" />}
      </div>

      <div className="flex-1 min-h-[100px] max-h-[300px] overflow-y-auto custom-scrollbar bg-black/20 rounded-lg p-4 text-sm leading-relaxed">
        {response ? (
          <div className="markdown-body prose prose-invert prose-sm max-w-none">
            <Markdown>{response.replace(/\[ACTION:.*?\]/g, "")}</Markdown>
          </div>
        ) : (
          <div className="text-neural-muted italic flex items-center gap-2">
            <Sparkles className="w-3 h-3" />
            Ask me anything about how this network learns...
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
          placeholder="e.g. How does the gradient flow backwards?"
          className="flex-1 bg-neural-bg border border-neural-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-neural-accent transition-all"
        />
        <button
          onClick={handleAsk}
          disabled={isLoading || !query.trim()}
          className="bg-neural-accent hover:bg-neural-accent/90 disabled:opacity-50 text-white p-2 rounded-lg transition-all"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default function App() {
  const [inputText, setInputText] = useState("The quick brown fox jumps over the lazy dog");
  const [tokens, setTokens] = useState<TokenData[]>([]);
  const [activation, setActivation] = useState<ActivationFunction>('relu');
  const [isForwarding, setIsForwarding] = useState(false);
  const [isBackpropping, setIsBackpropping] = useState(false);
  const [currentStep, setCurrentStep] = useState<'idle' | 'forward' | 'backprop'>('idle');

  const runSimulation = async () => {
    setCurrentStep('forward');
    setIsForwarding(true);
    await new Promise(r => setTimeout(r, 1000));
    setIsForwarding(false);
    
    setCurrentStep('backprop');
    setIsBackpropping(true);
    await new Promise(r => setTimeout(r, 1000));
    setIsBackpropping(false);
    setCurrentStep('idle');
  };

  const getStepDescription = () => {
    switch(currentStep) {
      case 'forward': return "Calculating activations: Input data is flowing through weights and activation functions...";
      case 'backprop': return "Learning from error: Gradients are flowing backwards to update the model's weights...";
      default: return "Currently Idle";
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col gap-6 max-w-7xl mx-auto">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl font-sans font-medium tracking-tight text-white mb-1">
            Neural <span className="text-neural-accent">Lens</span>
          </h1>
          <p className="text-neural-muted text-sm max-w-md">
            By Tanay Anand
            An interactive visualization of LLM Models
          </p>
        </div>
        
        <div className="flex items-center gap-3 glass-panel p-2 px-4">
          <div className="flex flex-col">
            <span className="mono-label">Activation</span>
            <div className="flex bg-neural-bg rounded-lg p-1 mt-1">
              {(['relu', 'sigmoid'] as ActivationFunction[]).map((func) => (
                <button
                  key={func}
                  onClick={() => setActivation(func)}
                  className={cn(
                    "px-3 py-1 text-xs font-mono rounded-md transition-all",
                    activation === func 
                      ? "bg-neural-accent text-white shadow-lg" 
                      : "text-neural-muted hover:text-white"
                  )}
                >
                  {func.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <div className="w-px h-8 bg-neural-border mx-2" />
          <button 
            onClick={runSimulation}
            disabled={isForwarding || isBackpropping}
            className="flex items-center gap-2 bg-neural-accent hover:bg-neural-accent/90 disabled:opacity-50 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all shadow-lg shadow-neural-accent/20"
          >
            {isForwarding || isBackpropping ? (
              <RotateCcw className="w-4 h-4 animate-spin" />
            ) : (
              <Zap className="w-4 h-4 fill-current" />
            )}
            {isForwarding ? "Forward Pass..." : isBackpropping ? "Backpropagating..." : "Run Simulation"}
          </button>
        </div>
      </header>

      {/* Status Bar */}
      <div className="glass-panel px-6 py-3 flex items-center gap-4 border-l-4 border-l-neural-accent">
        <div className="flex-shrink-0">
          {currentStep === 'idle' ? <Info className="w-5 h-5 text-neural-muted" /> : <Zap className="w-5 h-5 text-neural-accent animate-pulse" />}
        </div>
        <p className="text-sm font-mono text-neural-text">
          <span className="text-neural-accent mr-2">[{currentStep.toUpperCase()}]</span>
          {getStepDescription()}
        </p>
      </div>

      {/* Main Grid */}
      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        {/* Left Column: Input & Math */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="glass-panel p-6 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <Settings2 className="w-4 h-4 text-neural-accent" />
              <span className="mono-label">Input Configuration</span>
            </div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to tokenize..."
              className="w-full h-24 bg-neural-bg border border-neural-border rounded-lg p-3 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-neural-accent transition-all resize-none"
            />
            <div className="flex items-center justify-between text-[10px] text-neural-muted font-mono">
              <span>CHARS: {inputText.length}</span>
              <span>WORDS: {inputText.split(/\s+/).filter(Boolean).length}</span>
            </div>
          </div>

          <div className="h-[400px]">
            <MathPanel activation={activation} tokens={tokens} />
          </div>
        </div>

        {/* Right Column: Visualizations */}
        <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="md:col-span-1 h-[350px]">
            <Tokenizer text={inputText} onTokensChange={setTokens} />
          </div>
          <div className="md:col-span-1 h-[350px]">
            <VectorSpace tokens={tokens} />
          </div>
          <div className="md:col-span-2 h-[450px]">
            <NeuralNetworkVisualizer 
              activation={activation} 
              isForwarding={isForwarding} 
              isBackpropping={isBackpropping} 
            />
          </div>
          
          {/* Concept Deep Dive */}
          <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
            <AIChat onAction={(action) => {
              if (action === 'forward') runSimulation();
              if (action === 'backprop') runSimulation(); // Simplified for now
              if (action === 'relu') setActivation('relu');
              if (action === 'sigmoid') setActivation('sigmoid');
            }} />
            
            <div className="glass-panel p-6">
              <div className="flex items-center gap-2 mb-4">
                <Info className="w-4 h-4 text-neural-accent" />
                <span className="mono-label">Concept Deep Dive</span>
              </div>
              <div className="grid grid-cols-1 gap-6">
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-white flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-neural-accent" />
                    Embeddings
                  </h3>
                  <p className="text-xs text-neural-muted leading-relaxed">
                    Words are mapped to high-dimensional vectors. Similar words are placed closer together in this space, allowing the model to "understand" semantic relationships.
                  </p>
                </div>
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-white flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-neural-accent" />
                    Activation Functions
                  </h3>
                  <p className="text-xs text-neural-muted leading-relaxed">
                    {activation === 'relu' 
                      ? "ReLU (Rectified Linear Unit) is simple: it outputs the input if positive, else zero. It helps the network learn non-linear patterns efficiently."
                      : "Sigmoid squashes values between 0 and 1. It's historically significant but can suffer from 'vanishing gradients' in deep networks."}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer / Status */}
      <footer className="flex items-center justify-between border-t border-neural-border pt-4 px-2">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="mono-label"></span>
            <span className="text-[10px] font-mono text-neural-success uppercase tracking-widest"></span>
          </div>
          <div className="flex items-center gap-2">
            <span className="mono-label">Engine:</span>
            <span className="text-[10px] font-mono text-neural-muted uppercase tracking-widest"></span>
          </div>
        </div>
        <div className="text-[10px] font-mono text-neural-muted/50 uppercase">
          Neural Lens v1.0.0 - Tanay Anand
        </div>
      </footer>
    </div>
  );
}
