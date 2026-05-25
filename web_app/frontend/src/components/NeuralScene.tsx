import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial, Float } from '@react-three/drei';
import * as THREE from 'three';
import { MotionValue, useTransform } from 'framer-motion';

interface ParticleFieldProps {
  scrollProgress: MotionValue<number>;
}

const ParticleField: React.FC<ParticleFieldProps> = ({ scrollProgress }) => {
  const ref = useRef<THREE.Points>(null);
  const groupRef = useRef<THREE.Group>(null);
  
  // Generate random particles in a sphere
  const particles = useMemo(() => {
    const pos = new Float32Array(3000 * 3);
    for (let i = 0; i < 3000; i++) {
      const theta = THREE.MathUtils.randFloatSpread(360); 
      const phi = THREE.MathUtils.randFloatSpread(360); 
      const distance = 5 + Math.random() * 5;
      
      pos[i * 3] = distance * Math.sin(theta) * Math.cos(phi);
      pos[i * 3 + 1] = distance * Math.sin(theta) * Math.sin(phi);
      pos[i * 3 + 2] = distance * Math.cos(theta);
    }
    return pos;
  }, []);

  useFrame((state, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta / 20;
      ref.current.rotation.x += delta / 30;
    }
    
    if (groupRef.current) {
      // Dynamic rotation based on scroll
      const scroll = scrollProgress.get();
      groupRef.current.rotation.z = scroll * Math.PI;
      groupRef.current.position.z = scroll * 5;
    }
  });

  return (
    <group ref={groupRef}>
      <Points ref={ref} positions={particles} stride={3} frustumCulled={false}>
        <PointMaterial
          transparent
          color="#818cf8"
          size={0.03}
          sizeAttenuation={true}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          opacity={0.4}
        />
      </Points>
    </group>
  );
};

interface NeuralSceneProps {
  scrollProgress: MotionValue<number>;
}

const NeuralScene: React.FC<NeuralSceneProps> = ({ scrollProgress }) => {
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1, pointerEvents: 'none', background: '#020617' }}>
      <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
        <color attach="background" args={['#020617']} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#6366f1" />
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
          <ParticleField scrollProgress={scrollProgress} />
        </Float>
      </Canvas>
    </div>
  );
};

export default NeuralScene;

