import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldAlert, Newspaper, PieChart, Info, Search, Loader2, CheckCircle2, 
  XCircle, BarChart3, BrainCircuit, Activity, Globe, Zap, Cpu, Sparkles, 
  ChevronDown, Terminal, Server, ShieldCheck, Database, Layers, ExternalLink, ArrowRight,
  TrendingUp, MousePointer2, Box, Eye
} from 'lucide-react';
import { motion, AnimatePresence, useScroll, useSpring, useMotionValue, useTransform } from 'framer-motion';
import NeuralScene from './components/NeuralScene';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface PredictionResult {
  fake_news: { prediction: 'Real' | 'Fake'; confidence: number; model: string; };
  ai_detection: { prediction: 'Human' | 'AI Generated'; confidence: number; model: string; };
}

const SplashScreen = () => (
  <motion.div className="epic-splash" exit={{ opacity: 0, scale: 1.1 }} transition={{ duration: 1 }}>
    <div className="splash-wrap">
      <motion.div initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.8, type: "spring" }}>
        <ShieldAlert size={100} className="glow-icon" />
      </motion.div>
      <motion.h1 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>TRUTHLENS AI</motion.h1>
      <div className="splash-bar-outer"><motion.div className="splash-bar-inner" initial={{ width: 0 }} animate={{ width: "100%" }} transition={{ duration: 2.2 }} /></div>
    </div>
  </motion.div>
);

const App: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 100, damping: 30 });

  const tiltProps = {
    whileHover: { 
      rotateX: -5, rotateY: 5, scale: 1.02, 
      transition: { type: "spring", stiffness: 400, damping: 20 }
    },
    style: { perspective: 1000 }
  };

  useEffect(() => {
    const timer = setTimeout(() => setIsInitializing(false), 2800);
    return () => clearTimeout(timer);
  }, []);

  const handlePredict = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, model: 'deberta' }),
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || 'Neural processing failed');
      }

      if (!data.fake_news || !data.ai_detection) {
        throw new Error('Malformed response from neural node');
      }

      setResult(data);
      setTimeout(() => {
        document.getElementById('result-target')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 400);
    } catch (e: any) { 
      console.error(e);
      setError(e.message || 'An unexpected connection error occurred');
    } finally { 
      setLoading(false); 
    }
  };

  const analysisCards = [
    {
      title: "DeBERTa-v3 Architecture",
      tag: "CORE TECHNOLOGY",
      desc: "Our engine utilizes the Microsoft DeBERTa-v3 Transformer ensemble. Beyond just accuracy, it prioritizes semantic context, processing information with disentangled attention to understand subtle linguistic nuances. This design ensures the system isn't just matching keywords, but understanding the narrative structure of the news."
    },
    {
      title: "Balanced Neural Benchmarking",
      tag: "RELIABILITY",
      desc: "Performance is measured across a spectrum of metrics: Precision, Recall, and F1-Score. By optimizing for the harmonic mean of these values, we ensure a balanced detection system that minimizes False Alarms (protecting real news) while maintaining a strict filter against misinformation."
    },
    {
      title: "Synthetics & AI Detection",
      tag: "CONTENT ORIGIN",
      desc: "Trained on a specialized hybrid dataset of 24,000 samples, the model identifies the unique fingerprint of machine-generated text. It looks for over-optimization, a common trait in AI-generated fake news where the grammar is perfect but the semantic logic is manipulated."
    },
    {
      title: "Enterprise Efficiency",
      tag: "SYSTEM PERFORMANCE",
      desc: "High precision usually comes with high cost. Our architecture is engineered for rapid response times, achieving a golden ratio of deep-learning depth and operational speed. This allows for instantaneous news verification without server bottlenecks."
    },
    {
      title: "Semantic Pattern Mapping",
      tag: "INTELLIGENCE",
      desc: "The AI maps emotional triggers and sensationalist vocabulary. While factual reporting uses neutral descriptive tokens, misinformation patterns often rely on high-intensity emotional markers. Our system detects these patterns at a structural level, regardless of the topic."
    },
    {
      title: "Robust Data Integrity",
      tag: "FOUNDATION",
      desc: "Trust starts with the data. The model's foundation is a meticulously balanced split of verified real-world reporting and sophisticated fabrications. This rigorous training ensures the AI generalizes across global news events with zero historical bias."
    }
  ];

  return (
    <>
      <NeuralScene scrollProgress={scrollYProgress} />
      <AnimatePresence>{isInitializing && <SplashScreen />}</AnimatePresence>
      
      <div className="main-wrapper">
        <header className="nav-bar">
          <div className="nav-container">
            <div className="logo-group"><ShieldAlert className="logo-icon" /><span>TRUTHLENS AI</span></div>
            <nav className="links"><a href="#hero">Overview</a><a href="#verify">Verify</a><a href="#analysis">Insights</a><a href="#architecture">Systems</a></nav>
            <div className="status"><span className="dot" /> SYSTEM ONLINE</div>
          </div>
          <motion.div className="scroll-line" style={{ scaleX }} />
        </header>

        <section id="hero" className="hero-section full-page">
          <motion.div className="hero-content" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} transition={{ duration: 1 }}>
            <div className="badge-modern"><Sparkles size={14} /> <span>Neural Verification Framework</span></div>
            <h1>Illuminate the <span className="text-glow">Hidden Truth</span> <br/> in Global Media.</h1>
            <p>Next-generation artificial intelligence designed to deconstruct misinformation and identify synthetic content with laboratory-grade precision.</p>
            <div className="hero-btns">
              <motion.a href="#verify" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="btn-main">Get Started <ArrowRight size={18} /></motion.a>
              <a href="#analysis" className="btn-ghost">Research Data</a>
            </div>
          </motion.div>
        </section>

        <section id="verify" className="verify-section full-page">
          <div className="container">
            <motion.div className="section-title" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
              <h2>Verification Engine</h2>
              <p>Execute neural analysis to verify veracity and content origin</p>
            </motion.div>

            <div className="verify-grid">
              <motion.div 
                {...tiltProps}
                className="input-box glass-card" 
                initial={{ opacity: 0, x: -80 }} 
                whileInView={{ opacity: 1, x: 0 }} 
                viewport={{ once: true }} 
                transition={{ duration: 0.8, type: "spring", damping: 15 }}
              >
                <div className="card-header"><Terminal size={14} /> <span>INPUT_STREAM_RAW</span></div>
                <textarea 
                  placeholder="Paste news content here for deep analysis... (Note: Model optimized for political news context)" 
                  value={inputText} 
                  onChange={e=>setInputText(e.target.value)} 
                />
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <Info size={12} />
                  <span>Disclaimer: Training data is primarily focused on political issues. Results may vary for other topics.</span>
                </div>
                <div className="engine-select">
                  <label>ACTIVE NEURAL ARCHITECTURE: DEBERTA-V3 TRANSFORMER</label>
                </div>
                <button className="scan-btn" onClick={handlePredict} disabled={loading||!inputText.trim()}>
                  {loading ? <Loader2 className="spin"/> : <><Zap size={18}/> EXECUTE NEURAL SCAN</>}
                </button>
              </motion.div>

              <motion.div className="result-box" initial={{ opacity: 0, x: 80 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 0.8, type: "spring", damping: 15 }}>
                <div id="result-target" />
                <AnimatePresence mode="wait">
                  {error ? (
                    <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="error-display glass-card">
                      <ShieldAlert size={50} color="#ff4444" />
                      <h3>Neural Link Failure</h3>
                      <p>{error}</p>
                      <button onClick={() => setError(null)} className="retry-btn">RESET_PROTOCOL</button>
                    </motion.div>
                  ) : result ? (
                    <motion.div initial={{ opacity:0, scale:0.95 }} animate={{ opacity:1, scale:1 }} className="result-display">
                      <motion.div {...tiltProps} className={`res-item ${result.fake_news.prediction.toLowerCase()}`}>
                        <div className="res-tag-badge">VERACITY ANALYSIS</div>
                        <div className="res-row">
                          <div className="res-icon">{result.fake_news.prediction === 'Real' ? <CheckCircle2 size={36}/> : <XCircle size={36}/>}</div>
                          <div className="res-info">
                            <h3>{result.fake_news.prediction} News</h3>
                            <div className="conf-bar-wrap">
                              <div className="conf-labels"><span>Confidence</span><span>{result.fake_news.confidence}%</span></div>
                              <div className="conf-track"><motion.div className="conf-fill" initial={{width:0}} animate={{width:`${result.fake_news.confidence}%`}} transition={{duration:1}} /></div>
                            </div>
                          </div>
                        </div>
                      </motion.div>

                      <motion.div {...tiltProps} className={`res-item ${result.ai_detection.prediction === 'Human' ? 'real' : 'fake'}`} style={{marginTop:'20px'}}>
                        <div className="res-tag-badge">ORIGIN_DETECTION_LOG</div>
                        <div className="res-row">
                          <div className="res-icon">{result.ai_detection.prediction === 'Human' ? <Globe size={36}/> : <BrainCircuit size={36}/>}</div>
                          <div className="res-info">
                            <h3>{result.ai_detection.prediction} Content</h3>
                            <div className="conf-bar-wrap">
                              <div className="conf-labels"><span>Probability</span><span>{result.ai_detection.confidence}%</span></div>
                              <div className="conf-track"><motion.div className="conf-fill" initial={{width:0}} animate={{width:`${result.ai_detection.confidence}%`}} transition={{duration:1, delay:0.2}} /></div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    </motion.div>
                  ) : (
                    <div className="result-idle glass-card">
                      <div className="laser-scanner" />
                      <BrainCircuit size={80} className="idle-icon" />
                      <h3>Neural Standby</h3>
                      <p>Awaiting valid text input for pattern matching</p>
                    </div>
                  )}
                </AnimatePresence>
              </motion.div>
            </div>
          </div>
        </section>

        <section id="analysis" className="analysis-section full-page">
          <div className="container">
            <motion.div className="section-title" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h2>Deep Intelligence Insights</h2>
              <p>Understanding the multidimensional capabilities of our neural framework</p>
            </motion.div>
            <div className="analysis-grid">
              {analysisCards.map((card, i) => (
                <motion.div 
                  key={i} 
                  {...tiltProps}
                  className="analysis-card glass-card"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                >
                  <div className="card-label">{card.tag}</div>
                  <h3>{card.title}</h3>
                  <p style={{ fontSize: '16px', lineHeight: '1.8' }}>{card.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section id="architecture" className="arch-section full-page">
          <div className="container">
            <motion.div className="section-title" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h2>Enterprise Architecture</h2>
              <p>The high-performance technology stack powering our verification pipeline</p>
            </motion.div>
            <div className="tech-grid">
              {[
                {icon:<Cpu/>,title:'Inference Node',desc:'Asynchronous FastAPI backend optimized for high-concurrency neural workloads.'},
                {icon:<Database/>,title:'Semantic Repository',desc:'Balanced research dataset containing 24,000+ verified news samples.'},
                {icon:<Layers/>,title:'Ensemble Pipeline',desc:'Multi-stage architecture designed for maximum linguistic precision.'},
                {icon:<ShieldCheck/>,title:'Security Protocol',desc:'Zero-trust verification pipeline ensuring content integrity and scan safety.'}
              ].map((t,i)=>(
                <motion.div key={i} {...tiltProps} className="tech-card glass-card" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
                  <div className="tech-icon-wrap">{t.icon}</div>
                  <h4>{t.title}</h4>
                  <p>{t.desc}</p>
                </motion.div>
              ))}
            </div>
            <footer className="footer-pro">
              <div className="f-row">
                <span>© 2026 TRUTHLENS AI SYSTEMS. ALL RIGHTS RESERVED.</span>
                <div className="f-tags">
                  <TrendingUp size={14} /> <span>v1.5.0 STABLE</span>
                  <span className="dot-green" /><span>ENCRYPTED_STREAM</span>
                </div>
              </div>
            </footer>
          </div>
        </section>
      </div>
    </>
  );
};

export default App;
