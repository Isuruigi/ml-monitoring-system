'use client'

import { useEffect, useState } from 'react'
import { Activity, TrendingUp, AlertTriangle, Database, Zap, BarChart3 } from 'lucide-react'
import MetricsOverview from '@/components/MetricsOverview'
import DriftMonitor from '@/components/DriftMonitor'
import PredictionPanel from '@/components/PredictionPanel'

export default function Home() {
    const [healthStatus, setHealthStatus] = useState<any>(null)

    useEffect(() => {
        fetch(`${process.env.NEXT_PUBLIC_API_URL}/health`)
            .then(res => res.json())
            .then(data => setHealthStatus(data))
            .catch(err => console.error('Health check failed:', err))
    }, [])

    return (
        <div className="min-h-screen relative">
            {/* Animated Background */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-0 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-[128px] opacity-20 animate-pulse-slow"></div>
                <div className="absolute top-0 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-[128px] opacity-20 animate-pulse-slow" style={{ animationDelay: '2s' }}></div>
                <div className="absolute -bottom-40 left-1/2 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-[128px] opacity-20 animate-pulse-slow" style={{ animationDelay: '4s' }}></div>
            </div>

            {/* Header */}
            <header className="relative glass-strong border-b border-white/10 sticky top-0 z-50">
                <div className="container mx-auto px-6 py-5">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                            <div className="relative">
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg blur opacity-75 animate-pulse"></div>
                                <div className="relative bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                                    <Activity className="w-6 h-6 text-white" />
                                </div>
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                                    ML Monitoring System
                                </h1>
                                <p className="text-xs text-gray-400">Real-time Model Intelligence</p>
                            </div>
                        </div>
                        <div className="flex items-center space-x-6">
                            <div className="flex items-center space-x-2">
                                <div className={`relative w-2.5 h-2.5 rounded-full ${healthStatus?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'}`}>
                                    <div className={`absolute inset-0 rounded-full ${healthStatus?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'} animate-ping opacity-75`}></div>
                                </div>
                                <span className="text-sm font-medium text-white">
                                    {healthStatus?.status === 'healthy' ? 'Online' : 'Offline'}
                                </span>
                            </div>
                            <button className="glass px-4 py-2 rounded-lg hover:glass-strong transition-all hover-lift text-sm font-medium text-white">
                                Settings
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="container mx-auto px-6 py-8 relative">
                <div className="space-y-8 animate-fade-in">
                    {/* Hero Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <StatCard
                            icon={<TrendingUp className="w-7 h-7" />}
                            title="Model Accuracy"
                            value="78.5%"
                            change="+2.3%"
                            positive={true}
                            gradient="from-blue-500 to-cyan-500"
                        />
                        <StatCard
                            icon={<Zap className="w-7 h-7" />}
                            title="Predictions Today"
                            value="1,247"
                            change="+156"
                            positive={true}
                            gradient="from-purple-500 to-pink-500"
                        />
                        <StatCard
                            icon={<AlertTriangle className="w-7 h-7" />}
                            title="Drift Score"
                            value="0.18"
                            change="-0.05"
                            positive={true}
                            gradient="from-green-500 to-emerald-500"
                        />
                        <StatCard
                            icon={<Database className="w-7 h-7" />}
                            title="Model Version"
                            value="v2.1.0"
                            change="Latest"
                            positive={true}
                            gradient="from-orange-500 to-red-500"
                        />
                    </div>

                    {/* Metrics and Drift */}
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                        <MetricsOverview />
                        <DriftMonitor />
                    </div>

                    {/* Prediction Panel */}
                    <PredictionPanel />
                </div>
            </main>

            {/* Footer */}
            <footer className="mt-16 relative glass-strong border-t border-white/10">
                <div className="container mx-auto px-6 py-6">
                    <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
                        <div className="flex items-center space-x-2">
                            <BarChart3 className="w-5 h-5 text-blue-400" />
                            <p className="text-sm text-gray-400">
                                ML Monitoring System <span className="text-blue-400">v1.0.0</span>
                            </p>
                        </div>
                        <div className="flex items-center space-x-6 text-xs text-gray-500">
                            <span>Production Ready</span>
                            <span>•</span>
                            <span>FastAPI + Next.js</span>
                            <span>•</span>
                            <span className="text-green-400">All Systems Operational</span>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    )
}

function StatCard({ icon, title, value, change, positive, gradient }: any) {
    return (
        <div className="group relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl" style={{ backgroundImage: `linear-gradient(135deg, var(--tw-gradient-stops))` }}></div>
            <div className="relative glass rounded-2xl p-6 hover-lift transition-all duration-300 border border-white/10 hover:border-white/20">
                <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${gradient} shadow-lg`}>
                        <div className="text-white">{icon}</div>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${positive ? 'bg-green-500/20 text-green-400 border border-green-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                        {change}
                    </div>
                </div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">{title}</h3>
                <p className="text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                    {value}
                </p>
            </div>
        </div>
    )
}
