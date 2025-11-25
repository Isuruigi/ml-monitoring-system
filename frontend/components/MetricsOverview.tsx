'use client'

import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { TrendingUp, Award, Target, Zap } from 'lucide-react'

export default function MetricsOverview() {
    const [metrics, setMetrics] = useState<any>(null)

    useEffect(() => {
        fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/monitoring/metrics?time_period=24h`, {
            headers: {
                'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'demo-key'
            }
        })
            .then(res => res.json())
            .then(data => setMetrics(data))
            .catch(err => console.error('Failed to fetch metrics:', err))
    }, [])

    const performanceData = [
        { time: '00:00', accuracy: 0.76, f1: 0.74, precision: 0.75 },
        { time: '04:00', accuracy: 0.78, f1: 0.76, precision: 0.77 },
        { time: '08:00', accuracy: 0.77, f1: 0.75, precision: 0.76 },
        { time: '12:00', accuracy: 0.79, f1: 0.77, precision: 0.78 },
        { time: '16:00', accuracy: 0.78, f1: 0.76, precision: 0.77 },
        { time: '20:00', accuracy: 0.80, f1: 0.78, precision: 0.79 },
        { time: '24:00', accuracy: 0.785, f1: 0.765, precision: 0.775 },
    ]

    return (
        <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl"></div>
            <div className="relative glass rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500">
                            <TrendingUp className="w-5 h-5 text-white" />
                        </div>
                        <h2 className="text-xl font-bold text-white">Model Performance</h2>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
                        <span className="text-xs text-gray-400">Live</span>
                    </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-4 gap-3 mb-6">
                    <MetricBadge
                        icon={<Award className="w-4 h-4" />}
                        label="Accuracy"
                        value={metrics?.accuracy?.toFixed(3) || '0.785'}
                        color="blue"
                    />
                    <MetricBadge
                        icon={<Target className="w-4 h-4" />}
                        label="F1 Score"
                        value={metrics?.f1_score?.toFixed(3) || '0.765'}
                        color="purple"
                    />
                    <MetricBadge
                        icon={<Zap className="w-4 h-4" />}
                        label="Precision"
                        value={metrics?.precision?.toFixed(3) || '0.775'}
                        color="green"
                    />
                    <MetricBadge
                        icon={<TrendingUp className="w-4 h-4" />}
                        label="Recall"
                        value={metrics?.recall?.toFixed(3) || '0.770'}
                        color="orange"
                    />
                </div>

                {/* Performance Chart */}
                <div className="glass-strong rounded-xl p-4">
                    <ResponsiveContainer width="100%" height={220}>
                        <AreaChart data={performanceData}>
                            <defs>
                                <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorF1" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
                            <YAxis stroke="#6b7280" domain={[0.7, 0.85]} fontSize={12} />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    borderRadius: '12px',
                                    backdropFilter: 'blur(10px)'
                                }}
                                labelStyle={{ color: '#f3f4f6', fontWeight: 'bold' }}
                            />
                            <Legend wrapperStyle={{ fontSize: '12px' }} />
                            <Area
                                type="monotone"
                                dataKey="accuracy"
                                stroke="#3b82f6"
                                strokeWidth={3}
                                fill="url(#colorAccuracy)"
                            />
                            <Area
                                type="monotone"
                                dataKey="f1"
                                stroke="#8b5cf6"
                                strokeWidth={3}
                                fill="url(#colorF1)"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    )
}

function MetricBadge({ icon, label, value, color }: any) {
    const colorClasses = {
        blue: 'from-blue-500 to-blue-600',
        purple: 'from-purple-500 to-purple-600',
        green: 'from-green-500 to-green-600',
        orange: 'from-orange-500 to-orange-600',
    }

    return (
        <div className="glass rounded-lg p-3 hover:glass-strong transition-all duration-300 hover-lift">
            <div className={`inline-flex p-1.5 rounded-md bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} mb-2`}>
                <div className="text-white">{icon}</div>
            </div>
            <p className="text-xs text-gray-400 mb-1">{label}</p>
            <p className="text-xl font-bold text-white">{value}</p>
        </div>
    )
}
