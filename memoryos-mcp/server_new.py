
import sys
import os
import json
import argparse
from typing import Any, Dict, Optional, List
# 确保当前目录在sys.path中，以便导入memoryos模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'memoryos'))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print(f"ERROR: Failed to import FastMCP. Exception: {e}", file=sys.stderr)
    print("请安装最新版本的MCP: uv add mcp", file=sys.stderr)
    sys.exit(1)

try:
    from memoryos import Memoryos
    from utils import get_timestamp
except ImportError as e:
    print(f"无法导入MemoryOS模块: {e}", file=sys.stderr)
    print("请确保项目结构正确，memoryos目录应包含所有必要文件", file=sys.stderr)
    sys.exit(1)

# MemoryOS实例 - 将在初始化时设置
memoryos_instance: Optional[Memoryos] = None

def init_memoryos(config_path: str) -> Memoryos:
    """初始化MemoryOS实例"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    required_fields = ['user_id', 'openai_api_key', 'data_storage_path']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置文件缺少必需字段: {field}")
    
    return Memoryos(
        user_id=config['user_id'],
        openai_api_key=config['openai_api_key'],
        data_storage_path=config['data_storage_path'],
        openai_base_url=config.get('openai_base_url'),
        assistant_id=config.get('assistant_id', 'default_assistant_profile'),
        short_term_capacity=config.get('short_term_capacity', 10),
        mid_term_capacity=config.get('mid_term_capacity', 2000),
        long_term_knowledge_capacity=config.get('long_term_knowledge_capacity', 100),
        retrieval_queue_capacity=config.get('retrieval_queue_capacity', 7),
        mid_term_heat_threshold=config.get('mid_term_heat_threshold', 5.0),
        llm_model=config.get('llm_model', 'gpt-4o-mini'),
        embedding_model_name=config.get('embedding_model_name', 'text-embedding-3-small'),
        embedding_api_key=config.get('embedding_api_key'),
        embedding_base_url=config.get('embedding_base_url'),
    )

# 创建FastMCP服务器实例
mcp = FastMCP("MemoryOS")

@mcp.tool()
def add_memory(user_input: str, agent_response: str, timestamp: Optional[str] = None, meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    向MemoryOS系统添加新的记忆（用户输入和助手回应的对话对）
    
    Args:
        user_input: 用户的输入或问题
        agent_response: 助手的回应
        timestamp: 时间戳（可选，格式：YYYY-MM-DD HH:MM:SS）
        meta_data: 可选的元数据（JSON对象）
    
    Returns:
        包含操作结果的字典
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return {
            "status": "error",
            "message": "MemoryOS is not initialized. Please check the configuration file."
        }
    
    try:
        if not user_input or not agent_response:
            return {
                "status": "error",
                "message": "user_input and agent_response are required"
            }
        
        memoryos_instance.add_memory(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=timestamp or get_timestamp(),
            meta_data=meta_data or {}
        )
        
        result = {
            "status": "success",
            "message": "Memory has been successfully added to MemoryOS",
            "timestamp": timestamp or get_timestamp(),
            "details": {
                "user_input_length": len(user_input),
                "agent_response_length": len(agent_response),
                "has_meta_data": meta_data is not None
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error adding memory: {str(e)}"
        }

@mcp.tool()
def retrieve_memory(query: str, relationship_with_user: str = "friend", style_hint: str = "", max_results: int = 10) -> Dict[str, Any]:
    """
    根据查询从MemoryOS检索相关的记忆和上下文信息，包括短期记忆、中期记忆和长期知识
    
    Args:
        query: 检索查询，描述要寻找的信息
        relationship_with_user: 与用户的关系类型（如：friend, assistant, colleague等）
        style_hint: 回应风格提示
        max_results: 返回的最大结果数量
    
    Returns:
        包含检索结果的字典，包括：
        - short_term_memory: 当前短期记忆中的所有QA对
        - retrieved_pages: 从中期记忆检索的相关页面
        - retrieved_user_knowledge: 从用户长期知识库检索的相关条目
        - retrieved_assistant_knowledge: 从助手知识库检索的相关条目
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return {
            "status": "error",
            "message": "MemoryOS is not initialized. Please check the configuration file."
        }
    
    try:
        if not query:
            return {
                "status": "error",
                "message": "query parameter is required"
            }
        
        # 使用retriever获取相关上下文
        retrieval_results = memoryos_instance.retriever.retrieve_context(
            user_query=query,
            user_id=memoryos_instance.user_id
        )
        
        # 获取短期记忆内容
        short_term_history = memoryos_instance.short_term_memory.get_all()
        
        # 获取用户画像
        user_profile = memoryos_instance.get_user_profile_summary()
        
        # 组织返回结果
        result = {
            "status": "success",
            "query": query,
            "timestamp": get_timestamp(),
            "user_profile": user_profile if user_profile and user_profile.lower() != "none" else "No detailed user profile",
            "short_term_memory": short_term_history,
            "short_term_count": len(short_term_history),
            "retrieved_pages": [{
                'user_input': page['user_input'],
                'agent_response': page['agent_response'],
                'timestamp': page['timestamp'],
                'meta_info': page['meta_info']
            } for page in retrieval_results["retrieved_pages"][:max_results]],

            "retrieved_user_knowledge": [{
                    'knowledge': k['knowledge'],
                    'timestamp': k['timestamp']
                } for k in retrieval_results["retrieved_user_knowledge"][:max_results]],

            "retrieved_assistant_knowledge": [{
                'knowledge': k['knowledge'],
                'timestamp': k['timestamp']
            } for k in retrieval_results["retrieved_assistant_knowledge"][:max_results]],
            
            # 添加总数统计字段
            "total_pages_found": len(retrieval_results["retrieved_pages"]),
            "total_user_knowledge_found": len(retrieval_results["retrieved_user_knowledge"]),
            "total_assistant_knowledge_found": len(retrieval_results["retrieved_assistant_knowledge"])
        }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving memory: {str(e)}"
        }

@mcp.tool()
def get_user_profile(include_knowledge: bool = True, include_assistant_knowledge: bool = False) -> Dict[str, Any]:
    """
    获取用户的画像信息，包括个性特征、偏好和相关知识
    
    Args:
        include_knowledge: 是否包括用户相关的知识条目
        include_assistant_knowledge: 是否包括助手知识库
    
    Returns:
        包含用户画像信息的字典
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return {
            "status": "error",
            "message": "MemoryOS is not initialized. Please check the configuration file."
        }
    
    try:
        # 获取用户画像
        user_profile = memoryos_instance.get_user_profile_summary()
        
        result = {
            "status": "success",
            "timestamp": get_timestamp(),
            "user_id": memoryos_instance.user_id,
            "assistant_id": memoryos_instance.assistant_id,
            "user_profile": user_profile if user_profile and user_profile.lower() != "none" else "No detailed user profile"
        }
        
        if include_knowledge:
            user_knowledge = memoryos_instance.user_long_term_memory.get_user_knowledge()
            result["user_knowledge"] = [
                {
                    "knowledge": item["knowledge"],
                    "timestamp": item["timestamp"]
                }
                for item in user_knowledge
            ]
            result["user_knowledge_count"] = len(user_knowledge)
        
        if include_assistant_knowledge:
            assistant_knowledge = memoryos_instance.get_assistant_knowledge_summary()
            result["assistant_knowledge"] = [
                {
                    "knowledge": item["knowledge"],
                    "timestamp": item["timestamp"]
                }
                for item in assistant_knowledge
            ]
            result["assistant_knowledge_count"] = len(assistant_knowledge)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting user profile: {str(e)}"
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MemoryOS MCP Server")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="配置文件路径 (默认: config.json)"
    )
    
    args = parser.parse_args()
    
    global memoryos_instance
    
    try:
        # 初始化MemoryOS
        memoryos_instance = init_memoryos(args.config)
        print(f"MemoryOS MCP Server 已启动，用户ID: {memoryos_instance.user_id}", file=sys.stderr)
        print(f"配置文件: {args.config}", file=sys.stderr)
        
        # 启动MCP服务器 - 使用stdio传输
        mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        print("服务器被用户中断", file=sys.stderr)
    except Exception as e:
        print(f"启动服务器时发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()