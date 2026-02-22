import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line,
    XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
    ComposedChart
} from 'recharts';
import { Clock, TrendingUp, AlertTriangle, Zap, Activity, RefreshCw } from 'lucide-react';
import { api } from '../api';

function CustomTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    return (
        <div className="glass-card p-3 text-xs border border-white/10">
            <p className="font-medium text-white/80 mb-1">{label}</p>
            {payload.map((p, i) => (
                <p key={i} style={{ color: p.color }} className="flex justify-between gap-4">
                    <span>{p.name}:</span>
                    <span className="font-mono">{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</span>
                </p>
            ))}
        </div>
    );
}

export default function TimelineView() {
    const [timeline, setTimeline] = useState([]);
    const [flashLoans, setFlashLoans] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const [hours, setHours] = useState(24);
    const [loading, setLoading] = useState(true);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const [tl, fl, al] = await Promise.allSettled([
                api.timelineData(hours),
                api.flashLoans(),
                api.alerts(50),
            ]);
            if (tl.status === 'fulfilled') setTimeline(tl.value.timeline || []);
            if (fl.status === 'fulfilled') setFlashLoans(fl.value.events || []);
            if (al.status === 'fulfilled') setAlerts(al.value.alerts || []);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }, [hours]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    // Aggregate stats
    const totalTx = timeline.reduce((s, t) => s + t.tx_count, 0);
    const totalValue = timeline.reduce((s, t) => s + t.total_value, 0);
    const totalAlerts = timeline.reduce((s, t) => s + t.alert_count, 0);
    const peakHour = timeline.reduce((max, t) => t.tx_count > (max?.tx_count || 0) ? t : max, null);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold">Timeline Analysis</h1>
                    <p className="text-sm text-white/50 mt-1">Transaction patterns and anomaly timeline</p>
                </div>
                <div className="flex gap-2">
                    {[6, 12, 24, 48, 72].map((h) => (
                        <button
                            key={h}
                            onClick={() => setHours(h)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${hours === h
                                ? 'bg-primary-600 text-white'
                                : 'bg-white/5 text-white/50 hover:bg-white/10'
                                }`}
                        >
                            {h}h
                        </button>
                    ))}
                    <button
                        onClick={fetchData}
                        className="p-2 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors"
                    >
                        <RefreshCw className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Summary stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                    { icon: Activity, label: 'Transactions', value: totalTx.toLocaleString(), color: 'text-cyber-blue' },
                    { icon: TrendingUp, label: 'Total Volume', value: `${totalValue.toFixed(2)} ETH`, color: 'text-primary-400' },
                    { icon: AlertTriangle, label: 'Alerts', value: totalAlerts, color: 'text-amber-400' },
                    { icon: Zap, label: 'Flash Loans', value: flashLoans.length, color: 'text-cyber-pink' },
                ].map(({ icon: Icon, label, value, color }, i) => (
                    <motion.div
                        key={label}
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="stat-card"
                    >
                        <Icon className={`w-5 h-5 ${color}`} />
                        <p className="text-xl font-bold mt-2">{value}</p>
                        <p className="text-[10px] text-white/40 uppercase tracking-wider">{label} ({hours}h)</p>
                    </motion.div>
                ))}
            </div>

            {/* Main timeline chart */}
            <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="glass-card p-6"
            >
                <h3 className="text-sm font-semibold text-white/70 mb-4">Transaction Volume & Alerts Over Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={timeline}>
                        <defs>
                            <linearGradient id="gradTx" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="hour" tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} tickLine={false} />
                        <YAxis yAxisId="left" tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} axisLine={false} />
                        <YAxis yAxisId="right" orientation="right" tick={{ fill: 'rgba(255,255,255,0.2)', fontSize: 10 }} axisLine={false} />
                        <Tooltip content={<CustomTooltip />} />
                        <Area yAxisId="left" type="monotone" dataKey="tx_count" name="Transactions" stroke="#6366f1" fill="url(#gradTx)" strokeWidth={2} />
                        <Bar yAxisId="right" dataKey="alert_count" name="Alerts" fill="#ef4444" radius={[3, 3, 0, 0]} opacity={0.7} />
                        <Line yAxisId="left" type="monotone" dataKey="total_value" name="Volume (ETH)" stroke="#06b6d4" strokeWidth={2} dot={false} />
                    </ComposedChart>
                </ResponsiveContainer>
            </motion.div>

            {/* Bottom panels */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Flash Loan Events */}
                <motion.div
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="glass-card p-5"
                >
                    <h3 className="text-sm font-semibold text-white/70 mb-4 flex items-center gap-2">
                        <Zap className="w-4 h-4 text-cyber-pink" />
                        Flash Loan Events
                    </h3>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                        {flashLoans.length === 0 ? (
                            <p className="text-white/30 text-xs py-6 text-center">No flash-loan activity detected</p>
                        ) : (
                            flashLoans.slice(0, 15).map((fl, i) => (
                                <div key={i} className="p-3 rounded-xl bg-white/3 text-xs space-y-1">
                                    <div className="flex items-center justify-between">
                                        <span className="font-mono text-white/60 truncate max-w-[200px]">{fl.wallet}</span>
                                        <span className="text-cyber-pink font-semibold">{fl.flash_loan_score.toFixed(0)} pts</span>
                                    </div>
                                    <div className="flex gap-4 text-white/30">
                                        <span>Block #{fl.block_number}</span>
                                        <span>↓ {fl.inflow_eth.toFixed(4)} ETH</span>
                                        <span>↑ {fl.outflow_eth.toFixed(4)} ETH</span>
                                    </div>
                                    <p className="text-white/25 mt-1">{fl.explanation}</p>
                                </div>
                            ))
                        )}
                    </div>
                </motion.div>

                {/* Alert Timeline */}
                <motion.div
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    className="glass-card p-5"
                >
                    <h3 className="text-sm font-semibold text-white/70 mb-4 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        Alert History
                    </h3>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                        {alerts.length === 0 ? (
                            <p className="text-white/30 text-xs py-6 text-center">No alerts generated yet</p>
                        ) : (
                            alerts.slice(0, 15).map((alert) => (
                                <div key={alert.id} className="flex items-start gap-3 p-2.5 rounded-lg hover:bg-white/[0.03] transition-colors">
                                    <span className={`risk-${alert.severity} mt-0.5`}>{alert.severity}</span>
                                    <div className="flex-1 min-w-0">
                                        <p className="text-xs font-mono text-white/60 truncate">{alert.wallet_address}</p>
                                        <p className="text-[10px] text-white/30 mt-0.5">{alert.alert_type} • Score: {alert.risk_score.toFixed(0)}</p>
                                    </div>
                                    <span className="text-[10px] text-white/20">{alert.created_at?.slice(11, 16)}</span>
                                </div>
                            ))
                        )}
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
