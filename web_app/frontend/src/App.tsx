import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, Newspaper, Info, Loader2, CheckCircle2, 
  XCircle, BrainCircuit, Activity, Globe, Zap, Cpu, Sparkles, 
  Terminal, ShieldCheck, Database, Layers, ArrowRight,
  TrendingUp
} from 'lucide-react';
import { motion, AnimatePresence, useScroll, useSpring } from 'framer-motion';
import NeuralScene from './components/NeuralScene';

// Point frontend to our Hugging Face Space API as the default backend
const API_BASE_URL = (import.meta.env.VITE_API_URL || 'https://pyroblizzed-truthlensai.hf.space').replace(/\/$/, '');

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
  const [inputTitle, setInputTitle] = useState('');
  const [inputContent, setInputContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [scanHistory, setScanHistory] = useState<any[]>([]);
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
    
    // Load scan history from local storage
    const saved = localStorage.getItem('scan_history');
    if (saved) {
      try {
        setScanHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history:", e);
      }
    }
    
    return () => clearTimeout(timer);
  }, []);

  const handlePredict = async () => {
    if (!inputTitle.trim() || !inputContent.trim()) {
      setError("Please fill in both title and content for accurate analysis.");
      return;
    }
    console.log("Starting scan... Target URL:", API_BASE_URL);
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          title: inputTitle,
          text: inputContent, 
          model: 'bilstm' // Updated default model from deberta to bilstm
        }),
      });
      
      console.log("Response received. Status:", res.status);
      const data = await res.json();
      console.log("Data parsed:", data);
      
      if (!res.ok) {
        throw new Error(data.detail || `Server error: ${res.status}`);
      }

      setResult(data);
      
      // Save item to scan history
      const newHistoryItem = {
        id: Math.random().toString(36).substring(2, 9),
        title: inputTitle,
        text: inputContent.substring(0, 100) + (inputContent.length > 100 ? '...' : ''),
        date: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        fake_news: data.fake_news,
        ai_detection: data.ai_detection
      };
      
      setScanHistory(prev => {
        const updated = [newHistoryItem, ...prev.slice(0, 9)];
        localStorage.setItem('scan_history', JSON.stringify(updated));
        return updated;
      });

      setTimeout(() => {
        const target = document.getElementById('result-target');
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 500);
    } catch (e: any) { 
      console.error("Scan failed error:", e);
      setError(e.message || 'An unexpected connection error occurred');
    } finally { 
      setLoading(false); 
    }
  };

  const clearHistory = () => {
    setScanHistory([]);
    localStorage.removeItem('scan_history');
  };

  const analysisCards = [
    {
      title: "Bi-LSTM Neural Network",
      tag: "CORE MODEL",
      desc: "Our model uses a Bidirectional LSTM (Long Short-Term Memory) network paired with an attention pooling layer. It reads text forwards and backwards to capture full contextual relationships between words. It is highly optimized, fast, and does not require heavy GPU compute resources to run."
    },
    {
      title: "Balanced Training Metrics",
      tag: "PERFORMANCE",
      desc: "We track Precision, Recall, and F1-Score to ensure our model doesn't just guess one class. By optimizing the model for a high F1-Score, we keep false alarms to a minimum, ensuring real news is protected while successfully filtering out fabrications."
    },
    {
      title: "WELFake Dataset Foundation",
      tag: "DATASET",
      desc: "The classifier was trained on the public WELFake dataset, which contains over 72,000 verified real and fake news articles. This huge variety of data exposes the model to diverse writing styles, topics, and structures across global journalism."
    },
    {
      title: "Sub-10ms Inference Time",
      tag: "LATENCY",
      desc: "Unlike heavy modern transformers that cause server bottlenecks and latency, the Bi-LSTM model is lightweight and executes in under 10 milliseconds. This makes it ideal for real-time applications and public deployments."
    },
    {
      title: "Word Pattern Profiling",
      tag: "FEATURE EXTRACTION",
      desc: "Our model recognizes linguistic patterns typical of misinformation. While reliable news sources lean on neutral vocabulary and descriptive facts, fake articles often feature high-intensity emotional markers, clickbait wording, and aggressive formatting."
    },
    {
      title: "Open Code & Transparency",
      tag: "OPEN SOURCE",
      desc: "Everything is open and verifiable. The data preprocessing scripts, pipeline notebooks, model architectures, and training weights are documented in our repository, making it easy for anyone to inspect, run, or audit the code."
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
            <h1>Spot Fake News Instantly <br/> with <span className="text-glow">Machine Learning</span>.</h1>
            <p>A simple, fast tool that uses a Bidirectional LSTM network to analyze news text and detect whether it is real or fake.</p>
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
              <p>Analyze articles to verify fake_real and content ai_human</p>
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
                
                <div className="input-group-modern">
                  <label><Newspaper size={12} /> ARTICLE TITLE</label>
                  <input 
                    type="text"
                    placeholder="Enter headline here..."
                    value={inputTitle}
                    onChange={e=>setInputTitle(e.target.value)}
                    className="title-input-field"
                  />
                </div>

                <div className="input-group-modern" style={{ marginTop: '20px' }}>
                  <label><Terminal size={12} /> ARTICLE CONTENT</label>
                  <textarea 
                    placeholder="Paste news content here for deep analysis... (Note: Model optimized for political news context)" 
                    value={inputContent} 
                    onChange={e=>setInputContent(e.target.value)} 
                  />
                </div>

                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '5px', marginTop: '15px' }}>
                  <Info size={12} />
                  <span>Disclaimer: Training data is primarily focused on political issues. Results may vary for other topics.</span>
                </div>
                <div className="engine-select">
                  <label>ACTIVE NEURAL ARCHITECTURE: BI-LSTM CLASSIFIER</label>
                </div>
                <button className="scan-btn" onClick={handlePredict} disabled={loading||!inputTitle.trim()||!inputContent.trim()}>
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
                        <div className="res-tag-badge">ORIGIN DETECTION</div>
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
                      <h3>Ready to Scan</h3>
                      <p>Paste an article on the left and click check to start.</p>
                    </div>
                  )}
                </AnimatePresence>

                {/* Scan History Section */}
                {scanHistory.length > 0 && (
                  <motion.div 
                    initial={{ opacity: 0, y: 20 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    className="history-panel glass-card" 
                    style={{ marginTop: '20px', padding: '24px' }}
                  >
                    <div className="card-header" style={{ marginBottom: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Activity size={14} /> <span>SCAN_HISTORY_LOG</span>
                      </div>
                      <button 
                        onClick={clearHistory} 
                        className="clear-history-btn"
                        style={{ 
                          background: 'none', 
                          border: 'none', 
                          color: 'var(--e-danger)', 
                          fontSize: '11px', 
                          fontFamily: 'inherit',
                          fontWeight: 700, 
                          cursor: 'pointer',
                          opacity: 0.7
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
                        onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.7')}
                      >
                        CLEAR_LOGS
                      </button>
                    </div>
                    <div className="history-list" style={{ maxHeight: '200px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {scanHistory.map((item) => (
                        <div 
                          key={item.id} 
                          className="history-item" 
                          style={{ 
                            padding: '12px 16px', 
                            background: 'rgba(255,255,255,0.02)',
                            border: '1px solid rgba(255,255,255,0.05)',
                            borderRadius: '12px',
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center',
                            gap: '15px'
                          }}
                        >
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div className="history-item-title" style={{ fontWeight: 700, fontSize: '14px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#fff' }}>{item.title}</div>
                            <div className="history-item-date" style={{ fontSize: '10px', color: 'var(--e-text-muted)', marginTop: '4px' }}>{item.date}</div>
                          </div>
                          <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                            <span 
                              className={`badge-fake_real ${item.fake_news.prediction.toLowerCase()}`} 
                              style={{ 
                                fontSize: '10px', 
                                fontWeight: 800,
                                padding: '4px 8px', 
                                borderRadius: '6px', 
                                background: item.fake_news.prediction === 'Real' ? 'rgba(52, 211, 153, 0.1)' : 'rgba(251, 113, 133, 0.1)', 
                                color: item.fake_news.prediction === 'Real' ? 'var(--e-success)' : 'var(--e-danger)',
                                border: `1px solid ${item.fake_news.prediction === 'Real' ? 'rgba(52, 211, 153, 0.2)' : 'rgba(251, 113, 133, 0.2)'}`
                              }}
                            >
                              {item.fake_news.prediction.toUpperCase()}
                            </span>
                            <span 
                              className={`badge-ai_human ${item.ai_detection.prediction === 'Human' ? 'real' : 'fake'}`} 
                              style={{ 
                                fontSize: '10px', 
                                fontWeight: 800,
                                padding: '4px 8px', 
                                borderRadius: '6px', 
                                background: item.ai_detection.prediction === 'Human' ? 'rgba(129, 140, 248, 0.1)' : 'rgba(167, 139, 250, 0.1)', 
                                color: item.ai_detection.prediction === 'Human' ? 'var(--e-indigo)' : 'var(--e-violet)',
                                border: `1px solid ${item.ai_detection.prediction === 'Human' ? 'rgba(129, 140, 248, 0.2)' : 'rgba(167, 139, 250, 0.2)'}`
                              }}
                            >
                              {item.ai_detection.prediction === 'Human' ? 'HUMAN' : 'AI'}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
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
                {icon:<Database/>,title:'Semantic Repository',desc:'Balanced research dataset containing 72,000+ verified news samples.'},
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
