from collections import deque
from utils import get_timestamp
import heapq
class RetrievalAndAnswer:
    def __init__(self, short_term_memory, mid_term_memory, long_term_memory, dynamic_updater, queue_capacity=25):
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.dynamic_updater = dynamic_updater
        self.queue_capacity = queue_capacity
        self.retrieval_queue = deque(maxlen=queue_capacity)

    def retrieve(self, user_query, segment_threshold=0.7, page_threshold=0.7, knowledge_threshold=0.7, client=None):
            print("检索：开始检索中期记忆...")
            matched = self.mid_term_memory.search_sessions_by_summary(user_query, client, segment_threshold, page_threshold)
            
            # 使用堆来维护分数最高的页面
            top_pages_heap = []
            
            for item in matched:
                for page_info in item["matched_pages"]:  # 现在每个page_info是[page, overall]形式
                    page, overall_score = page_info
                    # 使用最小堆来保持前queue_capacity个最高分项目
                    if len(top_pages_heap) < self.queue_capacity:
                        heapq.heappush(top_pages_heap, (overall_score, id(page), page))
                    else:
                        # 如果当前分数高于堆中最小的分数，则替换
                        if overall_score > top_pages_heap[0][0]:
                            heapq.heappop(top_pages_heap)
                            heapq.heappush(top_pages_heap, (overall_score, id(page), page))
            
            # 清空并重新填充检索队列，按分数从高到低排序
            self.retrieval_queue.clear()
            for score, _, page in sorted(top_pages_heap, key=lambda x: x[0], reverse=True):
                self.retrieval_queue.append(page)
            
            print(f"检索：中期记忆召回 {len(self.retrieval_queue)} 个 QA 对到检索队列。")
            long_term_info = self.long_term_memory.search_knowledge(user_query, threshold=knowledge_threshold)
            # print(long_term_info[0].keys())
            print(f"检索：长期记忆召回 {len(long_term_info)} 个知识条目。")
            
            return {
                "retrieval_queue": list(self.retrieval_queue),
                "long_term_knowledge": long_term_info,
                "retrieved_at": get_timestamp()
            }
