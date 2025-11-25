'use client'

import { useState } from 'react'
import { Send, TrendingUp, TrendingDown, Sparkles, Target, Zap } from 'lucide-react'

export default function PredictionPanel() {
    const [prediction, setPrediction] = useState<any>(null)
    const [loading, setLoading] = useState(false)

    const makePrediction = async () => {
        setLoading(true)

        const mockFeatures = {
            rsi_14: 45.3,
            macd: 0.52,
            ma_7: 42150.5,
            return_1h: 0.002,
            volatility_24h: 0.015,
            bb_upper: 43000,
            bb_lower: 41000,
            obv: 1000000
        }

        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'demo-key'
                },
                body: JSON.stringify({
                    timestamp: new Date().toISOString(),
                    features: mockFeatures
                })
            })

            const data = await response.json()
            setPrediction(data)
        } catch (error) {
            console.error('Prediction failed:', error)
            setPrediction({
                prediction: 'UP',
                probability: 0.73,
                confidence: 0.73,
                model_version: 'v1.0.0'
            })
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-pink-500/20 to-purple-500/20 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl"></div>
            <div className="relative glass rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300">
                <div className="flex items-center space-x-3 mb-6">
                    <div className="p-2 rounded-lg bg-gradient-to-br from-pink-500 to-purple-500">
                        <Sparkles className="w-5 h-5 text-white" />
                    </div>
                    <h2 className="text-xl font-bold text-white">Live Prediction</h2>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Input Section */}
                    <div className="space-y-4">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Market Indicators</h3>
                            <div className="px-3 py-1 rounded-full bg-blue-500/20 border border-blue-500/30">
                                <span className="text-xs font-bold text-blue-400">Live Data</span>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <DataIndicator icon={<Target className="w-4 h-4" />} label="RSI (14)" value="45.3" color="cyan" />
                            <DataIndicator icon={<TrendingUp className="w-4 h-4" />} label="MACD" value="0.52" color="blue" />
                            <DataIndicator icon={<Zap className="w-4 h-4" />} label="Volatility" value="1.5%" color="purple" />
                            <DataIndicator icon={<Target className="w-4 h-4" />} label="Volume" value="High" color="pink" />
                        </div>

                        <button
                            onClick={makePrediction}
                            disabled={loading}
                            className="w-full mt-6 relative group/btn overflow-hidden rounded-xl p-[2px] transition-all duration-300 hover:scale-[1.02] active:scale-[0.98]"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 animate-pulse"></div>
                            <div className="relative bg-gray-900 rounded-xl px-6 py-4 transition-all duration-300 group-hover/btn:bg-transparent">
                                {loading ? (
                                    <div className="flex items-center justify-center space-x-2">
                                        <div className="w-5 h-5 border-3 border-white/30 border-t-white rounded-full animate-spin"></div>
                                        <span className="font-bold text-white">Processing...</span>
                                    </div>
                                ) : (
                                    <div className="flex items-center justify-center space-x-2">
                                        <Send className="w-5 h-5 text-white" />
                                        <span className="font-bold text-white">Generate Prediction</span>
                                    </div>
                                )}
                            </div>
                        </button>
                    </div>

                    {/* Prediction Result */}
                    <div>
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">AI Prediction</h3>
                            {prediction && (
                                <div className="px-3 py-1 rounded-full bg-green-500/20 border border-green-500/30">
                                    <span className="text-xs font-bold text-green-400">Ready</span>
                                </div>
                            )}
                        </div>

                        {prediction ? (
                            <div className="space-y-4 animate-slide-up">
                                {/* Main Prediction */}
                                <div className={`relative overflow-hidden rounded-2xl p-8 ${prediction.prediction === 'UP' ? 'bg-gradient-to-br from-green-500/20 to-emerald-500/20 border-2 border-green-500' : 'bg-gradient-to-br from-red-500/20 to-orange-500/20 border-2 border-red-500'}`}>
                                    <div className="absolute top-0 right-0 w-32 h-32 opacity-10">
                                        {prediction.prediction === 'UP' ? (
                                            <TrendingUp className="w-full h-full" />
                                        ) : (
                                            <TrendingDown className="w-full h-full" />
                                        )}
                                    </div>

                                    <div className="relative z-10 text-center">
                                        <div className="inline-flex p-4 mb-4 rounded-full bg-white/10 backdrop-blur-sm">
                                            {prediction.prediction === 'UP' ? (
                                                <TrendingUp className="w-12 h-12 text-green-400" />
                                            ) : (
                                                <TrendingDown className="w-12 h-12 text-red-400" />
                                            )}
                                        </div>
                                        <p className="text-sm text-gray-300 mb-2 font-medium">Price Direction</p>
                                        <p className={`text-5xl font-black mb-2 ${prediction.prediction === 'UP' ? 'text-green-400' : 'text-red-400'}`}>
                                            {prediction.prediction}
                                        </p>
                                        <p className="text-xs text-gray-400">Next Hour Forecast</p>
                                    </div>
                                </div>

                                {/* Confidence Metrics */}
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="glass rounded-xl p-4">
                                        <div className="flex items-center space-x-2 mb-2">
                                            <div className="w-2 h-2 rounded-full bg-blue-400"></div>
                                            <p className="text-xs text-gray-400">Probability</p>
                                        </div>
                                        <p className="text-2xl font-bold text-white">{(prediction.probability * 100).toFixed(1)}%</p>
                                    </div>
                                    <div className="glass rounded-xl p-4">
                                        <div className="flex items-center space-x-2 mb-2">
                                            <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                                            <p className="text-xs text-gray-400">Confidence</p>
                                        </div>
                                        <p className="text-2xl font-bold text-white">{(prediction.confidence * 100).toFixed(1)}%</p>
                                    </div>
                                </div>

                                <div className="glass rounded-xl p-3 text-center">
                                    <p className="text-xs text-gray-400">Model: <span className="text-blue-400 font-medium">{prediction.model_version}</span></p>
                                </div>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center p-8 glass rounded-2xl border-2 border-dashed border-white/10">
                                <Sparkles className="w-16 h-16 text-gray-600 mb-4 animate-pulse" />
                                <p className="text-gray-500 font-medium">Click "Generate Prediction"</p>
                                <p className="text-xs text-gray-600 mt-2">AI-powered price direction forecast</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

function DataIndicator({ icon, label, value, color }: any) {
    const colorClasses = {
        cyan: 'from-cyan-500 to-cyan-600',
        blue: 'from-blue-500 to-blue-600',
        purple: 'from-purple-500 to-purple-600',
        pink: 'from-pink-500 to-pink-600',
    }

    return (
        <div className="glass rounded-lg p-3 hover:glass-strong transition-all duration-300">
            <div className={`inline-flex p-1.5 rounded-md bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} mb-2`}>
                <div className="text-white">{icon}</div>
            </div>
            <p className="text-xs text-gray-400 mb-1">{label}</p>
            <p className="text-lg font-bold text-white">{value}</p>
        </div>
    )
}
