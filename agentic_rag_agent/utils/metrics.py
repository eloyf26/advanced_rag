"""
Performance Metrics and Monitoring for Agentic RAG Agent
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import psutil


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: Union[float, int]
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeSeriesMetric:
    """Time series metric with retention"""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    retention_hours: int = 24
    
    def add_point(self, value: Union[float, int], tags: Dict[str, str] = None):
        """Add a new data point"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
        self._cleanup_old_points()
    
    def _cleanup_old_points(self):
        """Remove points older than retention period"""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        while self.points and self.points[0].timestamp < cutoff:
            self.points.popleft()
    
    def get_recent_values(self, hours: int = 1) -> List[float]:
        """Get values from the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            point.value for point in self.points 
            if point.timestamp >= cutoff
        ]
    
    def get_statistics(self, hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of recent values"""
        values = self.get_recent_values(hours)
        
        if not values:
            return {'count': 0}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }


class MetricsCollector:
    """
    Comprehensive metrics collection for RAG system
    """
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus
        self.registry = CollectorRegistry() if enable_prometheus else None
        
        # Time series metrics
        self.time_series_metrics: Dict[str, TimeSeriesMetric] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': []
        }
        
        # Initialize Prometheus metrics if enabled
        if self.enable_prometheus:
            self._init_prometheus_metrics()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.prom_counters = {
            'queries_total': Counter(
                'rag_queries_total',
                'Total number of queries processed',
                ['query_type', 'status'],
                registry=self.registry
            ),
            'searches_total': Counter(
                'rag_searches_total',
                'Total number of searches performed',
                ['search_method', 'status'],
                registry=self.registry
            ),
            'embeddings_total': Counter(
                'rag_embeddings_total',
                'Total number of embeddings generated',
                ['model', 'status'],
                registry=self.registry
            ),
            'cache_hits_total': Counter(
                'rag_cache_hits_total',
                'Total number of cache hits',
                ['cache_type'],
                registry=self.registry
            ),
            'errors_total': Counter(
                'rag_errors_total',
                'Total number of errors',
                ['error_type', 'component'],
                registry=self.registry
            )
        }
        
        self.prom_histograms = {
            'query_duration': Histogram(
                'rag_query_duration_seconds',
                'Query processing duration',
                ['query_type'],
                registry=self.registry
            ),
            'search_duration': Histogram(
                'rag_search_duration_seconds',
                'Search duration',
                ['search_method'],
                registry=self.registry
            ),
            'embedding_duration': Histogram(
                'rag_embedding_duration_seconds',
                'Embedding generation duration',
                ['model'],
                registry=self.registry
            ),
            'reranking_duration': Histogram(
                'rag_reranking_duration_seconds',
                'Reranking duration',
                ['strategy'],
                registry=self.registry
            )
        }
        
        self.prom_gauges = {
            'active_queries': Gauge(
                'rag_active_queries',
                'Number of active queries',
                registry=self.registry
            ),
            'cache_size': Gauge(
                'rag_cache_size_bytes',
                'Cache size in bytes',
                ['cache_type'],
                registry=self.registry
            ),
            'system_cpu_usage': Gauge(
                'rag_system_cpu_usage_percent',
                'System CPU usage percentage',
                registry=self.registry
            ),
            'system_memory_usage': Gauge(
                'rag_system_memory_usage_percent',
                'System memory usage percentage',
                registry=self.registry
            )
        }
    
    def record_query(
        self, 
        query_type: str, 
        duration: float, 
        status: str = 'success',
        **tags
    ):
        """Record query metrics"""
        with self.lock:
            # Time series
            metric_name = f'query_{query_type}_duration'
            if metric_name not in self.time_series_metrics:
                self.time_series_metrics[metric_name] = TimeSeriesMetric(metric_name)
            
            self.time_series_metrics[metric_name].add_point(duration, tags)
            
            # Prometheus
            if self.enable_prometheus:
                self.prom_counters['queries_total'].labels(
                    query_type=query_type, 
                    status=status
                ).inc()
                self.prom_histograms['query_duration'].labels(
                    query_type=query_type
                ).observe(duration)
    
    def record_search(
        self, 
        method: str, 
        duration: float, 
        results_count: int,
        status: str = 'success',
        **tags
    ):
        """Record search metrics"""
        with self.lock:
            # Time series
            duration_metric = f'search_{method}_duration'
            count_metric = f'search_{method}_results'
            
            for metric_name, value in [(duration_metric, duration), (count_metric, results_count)]:
                if metric_name not in self.time_series_metrics:
                    self.time_series_metrics[metric_name] = TimeSeriesMetric(metric_name)
                self.time_series_metrics[metric_name].add_point(value, tags)
            
            # Prometheus
            if self.enable_prometheus:
                self.prom_counters['searches_total'].labels(
                    search_method=method,
                    status=status
                ).inc()
                self.prom_histograms['search_duration'].labels(
                    search_method=method
                ).observe(duration)
    
    def record_embedding_request(
        self,
        model: str,
        text_count: int,
        cache_hits: int,
        processing_time: float,
        status: str = 'success'
    ):
        """Record embedding generation metrics"""
        with self.lock:
            # Time series
            for metric_name, value in [
                (f'embedding_{model}_duration', processing_time),
                (f'embedding_{model}_text_count', text_count),
                (f'embedding_{model}_cache_hits', cache_hits)
            ]:
                if metric_name not in self.time_series_metrics:
                    self.time_series_metrics[metric_name] = TimeSeriesMetric(metric_name)
                self.time_series_metrics[metric_name].add_point(value)
            
            # Prometheus
            if self.enable_prometheus:
                self.prom_counters['embeddings_total'].labels(
                    model=model,
                    status=status
                ).inc()
                self.prom_histograms['embedding_duration'].labels(
                    model=model
                ).observe(processing_time)
                self.prom_counters['cache_hits_total'].labels(
                    cache_type='embedding'
                ).inc(cache_hits)
    
    def record_reranking_request(
        self,
        strategy: str,
        document_count: int,
        processing_time: float,
        top_k: int,
        status: str = 'success'
    ):
        """Record reranking metrics"""
        with self.lock:
            # Time series
            for metric_name, value in [
                (f'reranking_{strategy}_duration', processing_time),
                (f'reranking_{strategy}_document_count', document_count),
                (f'reranking_{strategy}_top_k', top_k)
            ]:
                if metric_name not in self.time_series_metrics:
                    self.time_series_metrics[metric_name] = TimeSeriesMetric(metric_name)
                self.time_series_metrics[metric_name].add_point(value)
            
            # Prometheus
            if self.enable_prometheus:
                self.prom_histograms['reranking_duration'].labels(
                    strategy=strategy
                ).observe(processing_time)
    
    def record_error(self, error_type: str, component: str, **context):
        """Record error occurrence"""
        with self.lock:
            # Time series
            metric_name = f'errors_{component}_{error_type}'
            if metric_name not in self.time_series_metrics:
                self.time_series_metrics[metric_name] = TimeSeriesMetric(metric_name)
            self.time_series_metrics[metric_name].add_point(1, context)
            
            # Prometheus
            if self.enable_prometheus:
                self.prom_counters['errors_total'].labels(
                    error_type=error_type,
                    component=component
                ).inc()
    
    def record_cache_event(self, cache_type: str, event_type: str, size_bytes: int = None):
        """Record cache-related events"""
        with self.lock:
            # Time series
            metric_name = f'cache_{cache_type}_{event_type}'
            if metric_name not in self.time_series_metrics:
                self.time_series_metrics[metric_name] = TimeSeriesMetric(metric_name)
            self.time_series_metrics[metric_name].add_point(1)
            
            # Prometheus
            if self.enable_prometheus:
                if event_type in ['hit', 'miss']:
                    self.prom_counters['cache_hits_total'].labels(
                        cache_type=cache_type
                    ).inc()
                
                if size_bytes is not None:
                    self.prom_gauges['cache_size'].labels(
                        cache_type=cache_type
                    ).set(size_bytes)
    
    def update_active_queries(self, count: int):
        """Update active query count"""
        if self.enable_prometheus:
            self.prom_gauges['active_queries'].set(count)
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self.lock:
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'period_hours': hours,
                'metrics': {}
            }
            
            for name, metric in self.time_series_metrics.items():
                stats = metric.get_statistics(hours)
                if stats['count'] > 0:
                    summary['metrics'][name] = stats
            
            # Add system metrics
            summary['system'] = self._get_system_metrics()
            
            return summary
    
    def get_query_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get query-specific performance metrics"""
        query_metrics = {}
        
        with self.lock:
            for name, metric in self.time_series_metrics.items():
                if name.startswith('query_') and name.endswith('_duration'):
                    query_type = name.replace('query_', '').replace('_duration', '')
                    stats = metric.get_statistics(hours)
                    if stats['count'] > 0:
                        query_metrics[query_type] = stats
        
        return query_metrics
    
    def get_search_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get search-specific performance metrics"""
        search_metrics = {}
        
        with self.lock:
            for name, metric in self.time_series_metrics.items():
                if name.startswith('search_'):
                    parts = name.split('_')
                    if len(parts) >= 3:
                        method = parts[1]
                        metric_type = '_'.join(parts[2:])
                        
                        if method not in search_metrics:
                            search_metrics[method] = {}
                        
                        stats = metric.get_statistics(hours)
                        if stats['count'] > 0:
                            search_metrics[method][metric_type] = stats
        
        return search_metrics
    
    def get_error_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error metrics and trends"""
        error_metrics = {
            'total_errors': 0,
            'error_rate': 0.0,
            'errors_by_type': {},
            'errors_by_component': {}
        }
        
        with self.lock:
            total_requests = 0
            
            for name, metric in self.time_series_metrics.items():
                if name.startswith('errors_'):
                    stats = metric.get_statistics(hours)
                    error_count = stats.get('count', 0)
                    error_metrics['total_errors'] += error_count
                    
                    # Parse error type and component
                    parts = name.replace('errors_', '').split('_')
                    if len(parts) >= 2:
                        component = parts[0]
                        error_type = '_'.join(parts[1:])
                        
                        if component not in error_metrics['errors_by_component']:
                            error_metrics['errors_by_component'][component] = 0
                        error_metrics['errors_by_component'][component] += error_count
                        
                        if error_type not in error_metrics['errors_by_type']:
                            error_metrics['errors_by_type'][error_type] = 0
                        error_metrics['errors_by_type'][error_type] += error_count
                
                elif name.startswith('query_'):
                    stats = metric.get_statistics(hours)
                    total_requests += stats.get('count', 0)
            
            # Calculate error rate
            if total_requests > 0:
                error_metrics['error_rate'] = error_metrics['total_errors'] / total_requests
        
        return error_metrics
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        if not self.enable_prometheus:
            return ""
        
        return generate_latest(self.registry).decode('utf-8')
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Update Prometheus gauges
            if self.enable_prometheus:
                self.prom_gauges['system_cpu_usage'].set(cpu_percent)
                self.prom_gauges['system_memory_usage'].set(memory.percent)
            
            return system_metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _start_background_monitoring(self):
        """Start background thread for system monitoring"""
        def monitor_system():
            while True:
                try:
                    self._get_system_metrics()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    print(f"Error in system monitoring: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        metrics_data = self.get_metrics_summary(hours=24)
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        elif format.lower() == 'prometheus':
            with open(path, 'w') as f:
                f.write(self.get_prometheus_metrics())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self.lock:
            self.time_series_metrics.clear()
            
            if self.enable_prometheus:
                # Prometheus metrics can't be reset, but we can clear the registry
                self.registry = CollectorRegistry()
                self._init_prometheus_metrics()


class PerformanceProfiler:
    """Performance profiler for detailed timing analysis"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_profiles = {}
        self.lock = threading.Lock()
    
    def start_profile(self, profile_id: str, operation: str) -> str:
        """Start profiling an operation"""
        with self.lock:
            profile_data = {
                'operation': operation,
                'start_time': time.time(),
                'checkpoints': []
            }
            self.active_profiles[profile_id] = profile_data
            return profile_id
    
    def add_checkpoint(self, profile_id: str, checkpoint_name: str):
        """Add a checkpoint to an active profile"""
        with self.lock:
            if profile_id in self.active_profiles:
                checkpoint_time = time.time()
                start_time = self.active_profiles[profile_id]['start_time']
                
                self.active_profiles[profile_id]['checkpoints'].append({
                    'name': checkpoint_name,
                    'elapsed_time': checkpoint_time - start_time,
                    'timestamp': checkpoint_time
                })
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End profiling and return results"""
        with self.lock:
            if profile_id not in self.active_profiles:
                return {}
            
            profile_data = self.active_profiles.pop(profile_id)
            end_time = time.time()
            total_duration = end_time - profile_data['start_time']
            
            # Record metrics
            self.metrics_collector.record_query(
                query_type='profiled',
                duration=total_duration,
                status='success'
            )
            
            return {
                'operation': profile_data['operation'],
                'total_duration': total_duration,
                'checkpoints': profile_data['checkpoints']
            }


class MetricsDashboard:
    """Simple metrics dashboard generator"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        metrics = self.metrics_collector.get_metrics_summary(hours=24)
        query_metrics = self.metrics_collector.get_query_performance_metrics(hours=24)
        search_metrics = self.metrics_collector.get_search_performance_metrics(hours=24)
        error_metrics = self.metrics_collector.get_error_metrics(hours=24)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG System Metrics Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px; }}
                .metric-title {{ font-weight: bold; color: #333; }}
                .metric-value {{ font-size: 1.2em; color: #007acc; }}
                .error {{ color: #d32f2f; }}
                .success {{ color: #388e3c; }}
            </style>
        </head>
        <body>
            <h1>RAG System Metrics Dashboard</h1>
            <p>Generated: {datetime.utcnow().isoformat()}</p>
            
            <h2>System Overview</h2>
            <div class="metric-card">
                <div class="metric-title">CPU Usage</div>
                <div class="metric-value">{metrics.get('system', {}).get('cpu_usage_percent', 'N/A')}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value">{metrics.get('system', {}).get('memory_usage_percent', 'N/A')}%</div>
            </div>
            
            <h2>Query Performance</h2>
            {self._generate_query_metrics_html(query_metrics)}
            
            <h2>Search Performance</h2>
            {self._generate_search_metrics_html(search_metrics)}
            
            <h2>Error Summary</h2>
            <div class="metric-card">
                <div class="metric-title">Total Errors (24h)</div>
                <div class="metric-value error">{error_metrics.get('total_errors', 0)}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Error Rate</div>
                <div class="metric-value error">{error_metrics.get('error_rate', 0):.2%}</div>
            </div>
            
        </body>
        </html>
        """
        
        return html
    
    def _generate_query_metrics_html(self, query_metrics: Dict[str, Any]) -> str:
        """Generate HTML for query metrics"""
        html = ""
        for query_type, stats in query_metrics.items():
            html += f"""
            <div class="metric-card">
                <div class="metric-title">{query_type.title()} Queries</div>
                <div>Count: <span class="metric-value">{stats.get('count', 0)}</span></div>
                <div>Avg Duration: <span class="metric-value">{stats.get('mean', 0):.2f}s</span></div>
                <div>P95: <span class="metric-value">{stats.get('p95', 0):.2f}s</span></div>
            </div>
            """
        return html
    
    def _generate_search_metrics_html(self, search_metrics: Dict[str, Any]) -> str:
        """Generate HTML for search metrics"""
        html = ""
        for method, metrics in search_metrics.items():
            duration_stats = metrics.get('duration', {})
            results_stats = metrics.get('results', {})
            
            html += f"""
            <div class="metric-card">
                <div class="metric-title">{method.title()} Search</div>
                <div>Searches: <span class="metric-value">{duration_stats.get('count', 0)}</span></div>
                <div>Avg Duration: <span class="metric-value">{duration_stats.get('mean', 0):.2f}s</span></div>
                <div>Avg Results: <span class="metric-value">{results_stats.get('mean', 0):.1f}</span></div>
            </div>
            """
        return html


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(enable_prometheus: bool = True) -> MetricsCollector:
    """Get singleton metrics collector instance"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(enable_prometheus)
    
    return _metrics_collector


def cleanup_metrics_collector():
    """Cleanup metrics collector on shutdown"""
    global _metrics_collector
    _metrics_collector = None