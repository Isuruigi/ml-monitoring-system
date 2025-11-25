'use client'

import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { AlertTriangle, CheckCircle, Shield, Activity } from 'lucide-react'

export default function DriftMonitor() {
    const [drift, setDrift] = useState<any>(null)

    useEffect(() => {
        fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/monitoring/drift`, {
            headers: {
                'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'demo-key'
            }
        })
            .then(res => res.json())
            .then(data => setDrift(data))
            .catch(err => console.error('Failed to fetch drift:', err))
    }, [])

    const driftData = [
        { feature: 'RSI', psi: 0.12 },
        { feature: 'MACD', psi: 0.08 },
        { feature: 'Volume', psi: 0.22 },
        { feature: 'MA_7', psi: 0.15 },
        { feature: 'Volatility', psi: 0.18 },
    ]

    const driftScore = drift?.drift_score || 0.18
    const isDriftDetected = driftScore > 0.20

    const getBarColor = (value: number) => {
        if (value > 0.20) return '#ef4444' // red
        if (value > 0.15) return '#f59e0b' // orange
        return '#10b981' // green
    }

    return (
        <div className="relative group">
            <div className={`absolute inset-0 ${isDriftDetected ? 'bg-gradient-to-r from-red-500/20 to-orange-500/20' : 'bg-gradient-to-r from-green-500/20 to-emerald-500/20'} rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl`}></div>
            <div className="relative glass rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded-lg bg-gradient-to-br ${isDriftDetected ? 'from-red-500 to-orange-500' : 'from-green-500 to-emerald-500'}`}>
                            <Shield className="w-5 h-5 text-white" />
                        </div>
                        <h2 className="text-xl font-bold text-white">Drift Detection</h2>
                    </div>
                    <div className="flex items-center space-x-2">
                        <Activity className="w-4 h-4 text-blue-400 animate-pulse" />
                        <span className="text-xs text-gray-400">Monitoring</span>
                    </div>
                </div>

                {/* Drift Status Card */}
                <div className={`relative overflow-hidden rounded-xl p-5 mb-6 ${isDriftDetected ? 'gradient-border glass' : 'gradient-border glass'}`}>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                            <div className={`p-3 rounded-full ${isDriftDetected ? 'bg-red-500/20' : 'bg-green-500/20'} animate-pulse`}>
                                {isDriftDetected ? (
                                    <AlertTriangle className="w-7 h-7 text-red-400" />
                                ) : (
                                    <CheckCircle className="w-7 h-7 text-green-400" />
                                )}
                            </div>
                            <div>
                                <p className="text-sm font-medium text-gray-400 mb-1">Overall Drift Score</p>
                                <p className={`text-4xl font-bold ${isDriftDetected ? 'text-red-400' : 'text-green-400'}`}>
                                    {driftScore.toFixed(3)}
                                </p>
                            </div>
                        </div>
                        <div className="text-right">
                            <div className={`px-4 py-2 rounded-full ${isDriftDetected ? 'bg-red-500/20 border border-red-500/30' : 'bg-green-500/20 border border-green-500/30'} mb-2`}>
                                <p className={`text-sm font-bold ${isDriftDetected ? 'text-red-400' : 'text-green-400'}`}>
                                    {isDriftDetected ? 'Drift Detected' : 'Stable'}
                                </p>
                            </div>
                            <p className="text-xs text-gray-500">
                                Threshold: 0.200
                            </p>
                        </div>
                    </div>
                </div>

                {/* Feature Drift Chart */}
                <div className="glass-strong rounded-xl p-4 mb-4">
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={driftData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="feature" stroke="#6b7280" fontSize={12} />
                            <YAxis stroke="#6b7280" domain={[0, 0.25]} fontSize={12} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    borderRadius: '12px',
                                    backdropFilter: 'blur(10px)'
                                }}
                                labelStyle={{ color: '#f3f4f6', fontWeight: 'bold' }}
                                cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
                            />
                            <Bar dataKey="psi" radius={[8, 8, 0, 0]}>
                                {driftData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={getBarColor(entry.psi)} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Recommendation */}
                <div className={`rounded-xl p-4 ${isDriftDetected ? 'glass border border-red-500/30' : 'glass border border-blue-500/30'}`}>
                    <div className="flex items-start space-x-3">
                        <div className={`p-2 rounded-lg ${isDriftDetected ? 'bg-red-500/20' : 'bg-blue-500/20'}`}>
                            <AlertTriangle className={`w-4 h-4 ${isDriftDetected ? 'text-red-400' : 'text-blue-400'}`} />
                        </div>
                        <div>
                            <p className={`text-sm font-medium ${isDriftDetected ? 'text-red-300' : 'text-blue-300'} mb-1`}>
                                System Recommendation
                            </p>
                            <p className="text-xs text-gray-400">
                                {drift?.recommendation || 'Monitor drift levels. Consider retraining if drift persists above threshold for extended periods.'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
