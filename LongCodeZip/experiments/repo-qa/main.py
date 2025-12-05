from code_compressor import CodeCompressor
from utility import COMMENT_QUERY, progress
from data import CACHE_DIR, get_repoqa_data
from compute_score import compute_score, save_json
# from llmlingua import PromptCompressor
from loguru import logger
from tree_sitter_languages import get_language, get_parser
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import os
from enum import Enum
from typing import List, Tuple, Dict
import warnings
from dataclasses import dataclass
import sys

class ChunkStrategy(Enum):
    FUNCTION_BASED = "function_based"
    SLIDING_WINDOW = "sliding_window"


# Language-specific chunk markers
CHUNK_MARKERS = {
    "python": ["class", "def"],
    "cpp": ["class", "struct", "void", "int", "bool", "double", "float", "char", "auto"],
    "java": ["class", "interface", "void", "int", "boolean", "double", "float", "char"],
    "typescript": ["class", "interface", "function", "const", "let", "var"],
    "rust": ["fn", "struct", "impl", "trait", "enum"],
    "go": ["func", "type", "struct", "interface"]
}

# all languages
# ALL_LANGUAGES = ["python", "cpp", "java", "typescript", "rust", "go"]

# Model context template
TEMPLATE = "instruction\ncode_context\ndescription\ninstruction"

INSTRUCTION = (
    "Based on the function description and code context,"
    " please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:"
)


@dataclass
class CodeChunk:
    """Represents a chunk of code with its embedding"""
    content: str
    start_line: int
    end_line: int
    embedding: torch.Tensor = None


class CodeChunker:
    def __init__(self, language: str, strategy: ChunkStrategy = ChunkStrategy.FUNCTION_BASED,
                 window_size: int = 20, overlap_size: int = 10):
        self.language = language
        self.parser = get_parser(language)
        self.strategy = strategy
        self.window_size = window_size
        self.overlap_size = overlap_size

    def _is_function_or_class_start(self, line: str) -> bool:
        """Check if line starts a new function or class definition"""
        line = line.strip()
        return any(line.startswith(marker) for marker in CHUNK_MARKERS[self.language])

    def _chunk_by_function(self, lines: List[str]) -> List[CodeChunk]:
        """Split code into chunks based on function/class definitions"""
        chunks = []
        current_chunk_lines = []
        current_start = 0

        for i, line in enumerate(lines):
            if self._is_function_or_class_start(line) and current_chunk_lines:
                # Store previous chunk
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(chunk_content, current_start, i-1))
                current_chunk_lines = []
                current_start = i
            current_chunk_lines.append(line)

        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(CodeChunk(chunk_content, current_start, len(lines)-1))

        return chunks

    def _chunk_by_sliding_window(self, lines: List[str]) -> List[CodeChunk]:
        """Split code into chunks using sliding window approach"""
        chunks = []

        # Handle case when code is shorter than window size
        if len(lines) <= self.window_size:
            return [CodeChunk('\n'.join(lines), 0, len(lines)-1)]

        # Create overlapping chunks
        start = 0
        while start < len(lines):
            end = min(start + self.window_size, len(lines))
            chunk_content = '\n'.join(lines[start:end])
            chunks.append(CodeChunk(chunk_content, start, end-1))

            # Move start position by (window_size - overlap_size)
            start += self.window_size - self.overlap_size

            # If remaining lines are less than window_size, adjust start to include them in last chunk
            if len(lines) - start < self.window_size:
                if len(lines) - start > self.overlap_size:  # Only if there's enough new content
                    chunk_content = '\n'.join(lines[start:])
                    chunks.append(CodeChunk(chunk_content, start, len(lines)-1))
                break

        return chunks

    def chunk_code(self, code: str) -> List[CodeChunk]:
        """Split code into chunks based on selected strategy"""
        lines = code.split('\n')

        if self.strategy == ChunkStrategy.FUNCTION_BASED:
            return self._chunk_by_function(lines)
        elif self.strategy == ChunkStrategy.SLIDING_WINDOW:
            return self._chunk_by_sliding_window(lines)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")


class RAGCompressor:
    def __init__(self, model_name: str = "microsoft/unixcoder-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def compute_embeddings(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Compute embeddings for code chunks"""
        for chunk in chunks:
            inputs = self.tokenizer(chunk.content, return_tensors="pt",
                                    truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use mean pooling
            chunk.embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return chunks

    def get_relevant_chunks(self,
                            query_embedding: torch.Tensor,
                            chunks: List[CodeChunk],
                            top_k: int = 5) -> List[CodeChunk]:
        """Get most relevant chunks based on cosine similarity"""
        similarities = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            sim = torch.cosine_similarity(query_embedding, chunk.embedding, dim=0)
            similarities.append((sim.item(), chunk))

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]


def compress_context(code_context: str,
                     target_function: str,
                     language: str,
                     rag_compressor: RAGCompressor,
                     chunker: CodeChunker) -> str:
    """Compress code context using RAG approach"""
    # Split into chunks
    chunks = chunker.chunk_code(code_context)

    # Get original token count
    original_tokens = len(rag_compressor.tokenizer.encode(code_context))

    # Log original context size
    logger.info(f"Original context: {code_context}")
    logger.info(f"Original token count: {original_tokens}")
    logger.info(f"Number of chunks: {len(chunks)}")

    # Compute embeddings for all chunks
    chunks = rag_compressor.compute_embeddings(chunks)

    # Get embedding for target function
    target_embedding = rag_compressor.model(
        **rag_compressor.tokenizer(target_function, return_tensors="pt",
                                   truncation=True, max_length=512).to(rag_compressor.device)
    ).last_hidden_state.mean(dim=1).squeeze()

    # Get most relevant chunks
    relevant_chunks = rag_compressor.get_relevant_chunks(target_embedding, chunks)

    # Combine relevant chunks
    compressed_context = "\n".join(chunk.content for chunk in relevant_chunks)

    # Get compressed token count
    compressed_tokens = len(rag_compressor.tokenizer.encode(compressed_context))

    # Log compression results
    logger.info(f"Compressed token count: {compressed_tokens}")
    logger.info(f"Token compression ratio: {compressed_tokens/original_tokens:.2%}")
    logger.info("Selected chunks:")
    for i, chunk in enumerate(relevant_chunks):
        logger.info(f"Chunk {i+1} (lines {chunk.start_line}-{chunk.end_line}):\n{chunk.content}\n")

    return compressed_context


# def compress_context_llm_lingua(compressor: PromptCompressor,
#                                 code_context: str,
#                                 target_function: str,
#                                 language: str,
#                                 target_token: int = 1000) -> str:
#     """Compress code context using LLMLingua approach"""
#     # Get original token count using LLMLingua's tokenizer
#     original_tokens = len(compressor.tokenizer.encode(code_context))

#     # replace the "<|endoftext|>" in the code if there is any
#     if "<|endoftext|>" in code_context:
#         logger.warning(f"Removing <|endoftext|> in code context: {code_context}")
#         code_context = code_context.replace("<|endoftext|>", "")

#     # Compress the prompt
#     logger.info(f"Compressing prompt with instruction: \n{INSTRUCTION}")
#     logger.info(f"Code context: \n{code_context}")
#     logger.info(f"Description: \n{target_function}")
#     compressed = compressor.compress_prompt(
#         code_context,
#         instruction=INSTRUCTION,
#         question=target_function + INSTRUCTION,
#         target_token=target_token
#     )

#     compressed_prompt = compressed['compressed_prompt']
#     logger.info(f"Compressed prompt: \n{compressed_prompt}")

#     # Get compressed token count
#     compressed_tokens = len(compressor.tokenizer.encode(compressed_prompt))

#     # Log compression results
#     logger.info(f"Original token count: {original_tokens}")
#     logger.info(f"LLMLingua compressed token count: {compressed_tokens}")
#     logger.info(f"Token compression ratio: {compressed_tokens/original_tokens:.2%}")

#     return compressed_prompt


# def compress_context_longllmlingua_chunks(compressor: PromptCompressor,
#                                           code_context: str,
#                                           target_function: str,
#                                           language: str,
#                                           target_token: int = 1000,
#                                           chunk_size: int = 80,
#                                           overlap: int = 40) -> str:
#     """Compress code context using LongLLMLingua chunks approach"""
#     # Get original token count using LLMLingua's tokenizer
#     original_tokens = len(compressor.tokenizer.encode(code_context))

#     # replace the "<|endoftext|>" in the code if there is any
#     if "<|endoftext|>" in code_context:
#         logger.warning(f"Removing <|endoftext|> in code context: {code_context}")
#         code_context = code_context.replace("<|endoftext|>", "")

#     # Split code into chunks for longllmlingua_chunks method
#     lines = code_context.split('\n')
#     chunks = []
#     for i in range(0, len(lines), chunk_size - overlap):
#         chunk = lines[i:i + chunk_size]
#         if chunk:
#             chunks.append('\n'.join(chunk))

#     # Compress the prompt using chunks
#     compressed = compressor.compress_prompt(
#         chunks,
#         instruction=INSTRUCTION,
#         question=target_function + INSTRUCTION,
#         target_token=target_token,
#         rank_method="longllmlingua"
#     )

#     compressed_prompt = compressed['compressed_prompt']
#     logger.info(f"Compressed prompt: \n{compressed_prompt}")

#     # Get compressed token count
#     compressed_tokens = len(compressor.tokenizer.encode(compressed_prompt))

#     # Log compression results
#     logger.info(f"Original token count: {original_tokens}")
#     logger.info(f"LongLLMLingua chunks compressed token count: {compressed_tokens}")
#     logger.info(f"Token compression ratio: {compressed_tokens/original_tokens:.2%}")

#     return compressed_prompt


def compress_context_code_compressor(compressor: CodeCompressor,
                                     code_context: str,
                                     target_function: str,
                                     language: str,
                                     target_ratio: float = 0.5,
                                     ppl_strategy: str = "default",
                                     condition_in_question: str = "default",
                                     rank_only: bool = False,
                                     use_iterative_compression: bool = True,
                                     use_line_level_filter: bool = True) -> str:
    """Compress code context using CodeCompressor approach
    
    Args:
        compressor: The CodeCompressor instance
        code_context: The code to compress
        target_function: The function description/query
        language: The programming language
        target_ratio: Compression ratio (0.0-1.0)
        ppl_strategy: Strategy for perplexity calculation
        condition_in_question: Conditioning mode for perplexity
        rank_only: If True, only rank and select functions without fine-grained compression
        use_iterative_compression: Whether to use token-level iterative compression
        use_line_level_filter: Whether to use line-level filtering
    """
    # replace the "<|endoftext|>" in the code if there is any
    if "<|endoftext|>" in code_context:
        logger.warning(f"Removing <|endoftext|> in code context: {code_context}")
        code_context = code_context.replace("<|endoftext|>", "")

    # Compress the code using CodeCompressor
    if rank_only:
        # When rank_only is True, we'll use the compress_code_file method
        logger.info("===== Rank-only mode =====")
        compressed = compressor.compress_code_file(
            code=code_context,
            query=target_function,
            instruction=INSTRUCTION,
            rate=target_ratio,
            language=language,
            rank_only=True
        )
    else:
        # For non-function chunk processing, use compress_code if not splitting by functions
        if not use_line_level_filter and not use_iterative_compression:
            logger.info("===== Simple truncation mode =====")
            # Simple truncation mode
            compressed = compressor.compress_code(
                code=code_context,
                query=target_function,
                instruction=INSTRUCTION,
                rate=target_ratio,
                use_line_level_filter=False,
                use_iterative_compression=False
            )
        elif use_line_level_filter and not use_iterative_compression:
            logger.info("===== Line-level filtering only =====")
            # Line-level filtering only
            compressed = compressor.compress_code(
                code=code_context,
                query=target_function,
                instruction=INSTRUCTION,
                rate=target_ratio,
                use_line_level_filter=True,
                use_iterative_compression=False
            )
        elif not use_line_level_filter and use_iterative_compression:
            logger.info("===== Token-level iterative compression only =====")
            # Token-level iterative compression only
            compressed = compressor.compress_code(
                code=code_context,
                query=target_function,
                instruction=INSTRUCTION,
                rate=target_ratio,
                use_line_level_filter=False,
                use_iterative_compression=True
            )
        else:
            # Full function-based splitting and compression
            logger.info("===== Full function-based splitting and compression =====")
            compressed = compressor.compress_code_file(
                code=code_context,
                query=target_function,
                instruction=INSTRUCTION,
                rate=target_ratio,
                language=language,
                rank_only=False,
                use_iterative_compression=use_iterative_compression
            )

    # Get compressed prompt from results
    if "compressed_prompt" in compressed:
        compressed_prompt = compressed["compressed_prompt"]
    else:
        compressed_prompt = compressed["output"]

    # Log compression results
    logger.info(f"Original token count: {compressed['original_tokens']}")
    logger.info(f"CodeCompressor compressed token count: {compressed['compressed_tokens']}")
    logger.info(f"Token compression ratio: {compressed['compressed_tokens']/compressed['original_tokens']:.2%}")

    return compressed_prompt

def evaluate_model_rag(
    model: str,
    code_context_size: int = 16 * 1024,
    max_new_tokens: int = 1024,
    result_dir: str = "results/rag_compressed_v1",
    languages: List[str] = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    chunk_strategy: str = "function_based",
    window_size: int = 20,
    overlap_size: int = 10,
    dataset_path: str = None,
    compression_method: str = "rag",
    llm_lingua_target_token: int = 1000,
    compression_ratio: float = 0.5,
    backend: str = "vllm",
    ppl_strategy: str = "default",
    condition_in_question: str = "default",
    compression_mode: str = "function_focus",
    rank_only: bool = False,
    use_iterative_compression: bool = False,
    use_line_level_filter: bool = False,
    compression_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4"
):
    # show the parameters of rank_only, use_iterative_compression, use_line_level_filter
    logger.info(f"Rank-only: {rank_only}")
    logger.info(f"Use iterative compression: {use_iterative_compression}")
    logger.info(f"Use line-level filter: {use_line_level_filter}")

    """Main evaluation function with compression method selection
    
    Args:
        model: Model name or path
        code_context_size: Model context size in tokens
        max_new_tokens: Maximum tokens to generate
        result_dir: Directory to save results
        languages: List of languages to evaluate
        tensor_parallel_size: Tensor parallel size for vLLM
        trust_remote_code: Trust remote code for tokenizer and model
        chunk_strategy: Chunking strategy ("function_based" or "sliding_window")
        window_size: Window size for sliding window strategy
        overlap_size: Overlap size for sliding window strategy
        dataset_path: Path to dataset file
        compression_method: Compression method 
            ("rag", "llm_lingua", "longllmlingua_chunks", "code_compressor", "mgcode_compressor", "original")
        llm_lingua_target_token: Target token count for LLMLingua
        compression_ratio: Compression ratio for CodeCompressor
        backend: Backend for inference ("vllm")
        ppl_strategy: Perplexity strategy for CodeCompressor
        condition_in_question: Condition in question for CodeCompressor
        compression_mode: Compression mode for MGCodeCompressor
        rank_only: If True, only rank and select functions without fine-grained compression
        use_iterative_compression: Whether to use token-level iterative compression for code_compressor
        use_line_level_filter: Whether to apply line-level filtering for code_compressor
        compression_model: Model name for LLMLingua and CodeCompressor
    """
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)

    # Add strategy to the output directory name
    strategy_str = f"_{compression_method}"
    if compression_method == "llm_lingua":
        strategy_str += f"_t{llm_lingua_target_token}"
    elif compression_method == "longllmlingua_chunks":
        strategy_str += f"_t{llm_lingua_target_token}_w{window_size}_o{overlap_size}"
    elif compression_method == "code_compressor":
        # Create a compression mode string based on settings
        cc_mode = []
        if rank_only:
            cc_mode.append("rank_only")
        else:
            if use_iterative_compression:
                cc_mode.append("iter")
            if use_line_level_filter:
                cc_mode.append("line")
        
        mode_str = "_".join(cc_mode) if cc_mode else "simple"
        strategy_str += f"_t{compression_ratio}_mode_{mode_str}"
    if chunk_strategy == "sliding_window":
        strategy_str += f"_w{window_size}_o{overlap_size}"

    context_size_dir = os.path.join(result_dir, f"ntoken_{code_context_size}{strategy_str}")
    os.makedirs(context_size_dir, exist_ok=True)

    model_output_path = os.path.join(
        context_size_dir,
        f"{model.replace('/', '_slash_')}.jsonl",
    )
    
    # Intermediate file to store compressed contexts
    compressed_contexts_path = os.path.join(
        context_size_dir,
        f"compressed_contexts_{model.replace('/', '_slash_')}.jsonl",
    )

    # Load cache from Qwen results
    cache_file = os.path.join("data", "Qwen_slash_Qwen2.5-7B-Instruct.jsonl") # previous data from running original RepoQA
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    with open(cache_file) as f:
        cache = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(cache)} examples from {cache_file}")
    logger.info(f"Using chunking strategy: {chunk_strategy}")
    if chunk_strategy == "sliding_window":
        logger.info(f"Window size: {window_size}, Overlap size: {overlap_size}")
    if compression_method == "llm_lingua":
        logger.info(f"Using LLMLingua compression with target tokens: {llm_lingua_target_token}")
    elif compression_method == "longllmlingua_chunks":
        logger.info(f"Using LongLLMLingua chunks compression with:")
        logger.info(f"  - Target tokens: {llm_lingua_target_token}")
        logger.info(f"  - Chunk size: {window_size}")
        logger.info(f"  - Overlap: {overlap_size}")
    elif compression_method == "code_compressor":
        logger.info(f"Using CodeCompressor with ratio: {compression_ratio}")
        logger.info(f"CodeCompressor settings:")
        logger.info(f"  - rank_only: {rank_only}")
        logger.info(f"  - use_iterative_compression: {use_iterative_compression}")
        logger.info(f"  - use_line_level_filter: {use_line_level_filter}")

    # Filter by languages if specified
    if languages:
        cache = [c for c in cache if c["language"] in languages]

    if dataset_path is not None:
        with open(dataset_path) as f:
            dataset = json.load(f)
    else:
        dataset = get_repoqa_data()

    # If results already exist, load and evaluate
    if os.path.exists(model_output_path) and os.path.getsize(model_output_path) > 0:
        logger.info(f"Loading {model_output_path} and evaluating")
        model_outputs = [json.loads(line) for line in open(model_output_path)]
        file_base, _ = os.path.splitext(model_output_path)
        result_path = file_base + "-SCORES.json"
        output_json = compute_score(
            model,
            dataset,
            model_outputs,
            True,  # Ignore comments since we're using compressed context
            result_dir=result_dir,
        )
        save_json(output_json, result_path)
        return

    # PHASE 1: Compress all contexts
    compressed_tasks = []
    
    # Initialize appropriate compressor based on compression method
    if compression_method in ["rag", "original"]:
        rag_compressor = RAGCompressor()
    else:
        rag_compressor = None

    # Initialize compressors if needed
    llm_lingua_compressor = None
    code_compressor = None
    if compression_method in ["llm_lingua", "longllmlingua_chunks"]:
        # llm_lingua_compressor = PromptCompressor(compression_model)
        pass
    elif compression_method == "code_compressor":
        code_compressor = CodeCompressor(compression_model)
    # Convert string strategy to enum
    try:
        chunk_strategy_enum = ChunkStrategy(chunk_strategy)
    except ValueError:
        raise ValueError(f"Invalid chunk strategy: {chunk_strategy}. "
                         f"Must be one of {[s.value for s in ChunkStrategy]}")
    
    # Check if compressed contexts already exist
    if os.path.exists(compressed_contexts_path) and os.path.getsize(compressed_contexts_path) > 0:
        logger.info(f"Loading pre-compressed contexts from {compressed_contexts_path}")
        with open(compressed_contexts_path) as f:
            compressed_tasks = [json.loads(line) for line in f]
    else:
        logger.info(f"Starting compression phase for {len(cache)} examples")
        # Process and compress each task
        for i, task in enumerate(tqdm(cache, desc="Compressing contexts")):
            # Make a copy of the original task
            compressed_task = dict(task)
            
            try:
                # Compression logic based on selected method
                if compression_method == "rag":
                    chunker = CodeChunker(
                        task["language"],
                        strategy=chunk_strategy_enum,
                        window_size=window_size,
                        overlap_size=overlap_size
                    )
                    compressed_context = compress_context(
                        task["code_context"],
                        task["description"],
                        task["language"],
                        rag_compressor,
                        chunker=chunker
                    )
                # elif compression_method == "llm_lingua":
                #     compressed_context = compress_context_llm_lingua(
                #         compressor=llm_lingua_compressor,
                #         code_context=task["code_context"],
                #         target_function=task["description"],
                #         language=task["language"],
                #         target_token=llm_lingua_target_token
                #     )
                # elif compression_method == "longllmlingua_chunks":
                #     compressed_context = compress_context_longllmlingua_chunks(
                #         compressor=llm_lingua_compressor,
                #         code_context=task["code_context"],
                #         target_function=task["description"],
                #         language=task["language"],
                #         target_token=llm_lingua_target_token,
                #         chunk_size=window_size,
                #         overlap=overlap_size
                #     )
                elif compression_method == "code_compressor":
                    compressed_context = compress_context_code_compressor(
                        compressor=code_compressor,
                        code_context=task["code_context"],
                        target_function=task["description"],
                        language=task["language"],
                        target_ratio=compression_ratio,
                        ppl_strategy=ppl_strategy,
                        condition_in_question=condition_in_question,
                        rank_only=rank_only,
                        use_iterative_compression=use_iterative_compression,
                        use_line_level_filter=use_line_level_filter
                    )
                elif compression_method == "original":
                    compressed_context = task["code_context"]
                else:
                    raise ValueError(f"Invalid compression method: {compression_method}")
                
                # Update task with compressed context
                compressed_task["code_context"] = compressed_context
                
                # Generate prompt
                if compression_method == "code_compressor":
                    compressed_task["prompt"] = compressed_context
                else:
                    prompt = ""
                    for key in task["template"].split("\n"):
                        prompt += compressed_task[key]
                    compressed_task["prompt"] = prompt
                    
            except Exception as e:
                logger.error(f"Error compressing item {i} of {len(cache)}: {e}")
                # Use original context if compression fails
                compressed_task["code_context"] = task["code_context"]
                prompt = ""
                for key in task["template"].split("\n"):
                    prompt += compressed_task[key]
                compressed_task["prompt"] = prompt
            
            compressed_tasks.append(compressed_task)
            
            # Save intermediate results periodically
            if (i + 1) % 10 == 0 or i == len(cache) - 1:
                with open(compressed_contexts_path, "w") as f_out:
                    for t in compressed_tasks:
                        f_out.write(json.dumps(t) + "\n")
                        f_out.flush()
                logger.info(f"Saved {i+1}/{len(cache)} compressed contexts")
    
    # Clean up compressor objects to free memory
    del rag_compressor
    del llm_lingua_compressor
    del code_compressor
    
    # Force garbage collection to free GPU memory
    import gc
    gc.collect()
    
    # Clear CUDA cache if torch is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory cache")
    
    # PHASE 2: Generate responses with vLLM
    logger.info("Starting response generation phase")
    
    # Initialize vLLM provider
    from provider.vllm import VllmProvider
    engine = VllmProvider(
        model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=int(code_context_size * 1.5),
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=0.8  # Can use higher utilization now
    )
    
    # Generate responses for all compressed tasks
    model_outputs = []
    for i, task in enumerate(tqdm(compressed_tasks, desc="Generating responses")):
        # Generate reply
        replies = engine.generate_reply(
            task["prompt"], n=1, max_tokens=max_new_tokens
        )
        
        # Save result
        result = {**task, "output": replies}
        model_outputs.append(result)
    
    # Save all model outputs
    with open(model_output_path, "w") as f_out:
        for r in model_outputs:
            f_out.write(json.dumps(r) + "\n")
            f_out.flush()
    logger.info(f"Saved {len(model_outputs)} responses")

    # Compute and save scores
    file_base, _ = os.path.splitext(model_output_path)
    result_path = file_base + "-SCORES.json"
    output_json = compute_score(
        model,
        dataset,
        model_outputs,
        True,  # Ignore comments since we're using compressed context
        result_dir=result_dir,
    )
    save_json(output_json, result_path)


def main():
    from fire import Fire
    Fire(evaluate_model_rag)


if __name__ == "__main__":
    main()
