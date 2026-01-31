"""
Email and URL analysis endpoints.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Any
import asyncio

from fastapi import APIRouter, HTTPException, Depends
from pydantic import ValidationError

from app.schemas.requests import EmailAnalysisRequest, URLAnalysisRequest, BatchAnalysisRequest
from app.schemas.responses import AnalysisResponse, BatchAnalysisResponse
from app.schemas.enums import Verdict, RiskLevel, ModelType
from app.services.url_analyzer import url_analyzer
from app.services.feature_extractor import feature_extraction_service
from app.services.cache import cache_service
from app.services.model_service import get_xgboost_model, get_transformer_model, predict_with_xgboost, predict_with_transformer, predict_with_multi_agent
from app.middleware.metrics import record_model_prediction, record_cache_hit, record_cache_miss
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _generate_email_id() -> str:
    """Generate unique email ID."""
    return f"email_{uuid.uuid4().hex[:12]}"


def _risk_score_to_level(risk_score: int) -> RiskLevel:
    """Convert risk score to risk level."""
    if risk_score >= 80:
        return RiskLevel.CRITICAL
    elif risk_score >= 60:
        return RiskLevel.HIGH
    elif risk_score >= 40:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


@router.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """
    Quick URL-only analysis.

    Faster than full email analysis, focuses on URL-based detection.
    Uses heuristic analysis without ML models for sub-100ms response times.
    """
    email_id = _generate_email_id()
    start_time = datetime.now()

    try:
        # Check cache first
        cache_key = None
        if request.use_cache:
            cache_key = cache_service.generate_url_key(request.url)
            cached_result = await cache_service.get(cache_key)

            if cached_result:
                logger.info(f"URL cache hit: {request.url}")
                record_cache_hit("url_reputation")
                return AnalysisResponse(**cached_result)

            record_cache_miss("url_reputation")

        # Analyze URL
        result = await url_analyzer.analyze_url(
            url=request.url,
            context=request.context
        )

        # Map to response format
        response = AnalysisResponse(
            email_id=email_id,
            verdict=Verdict(result["verdict"]),
            confidence=result["risk_score"] / 100.0,
            risk_score=result["risk_score"],
            risk_level=_risk_score_to_level(result["risk_score"]),
            model_used="url_heuristic",
            analysis={
                "url_risk": {
                    "url": request.url,
                    "checks": result["checks"],
                    "risk_factors": [r for r in result["explanation"].split(". ") if r]
                }
            },
            explanation=result["explanation"],
            processing_time_ms=result["processing_time_ms"],
            cache_hit=False,
            timestamp=datetime.utcnow().isoformat()
        )

        # Cache result
        if request.use_cache and cache_key:
            await cache_service.set(
                cache_key,
                response.model_dump(),
                ttl=settings.REDIS_URL_REPUTATION_TTL
            )

        # Record metrics
        record_model_prediction(
            model_type="url_heuristic",
            verdict=result["verdict"],
            duration_ms=result["processing_time_ms"]
        )

        return response

    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"URL analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="URL analysis failed")


@router.post("/analyze/email", response_model=AnalysisResponse)
async def analyze_email(request: EmailAnalysisRequest):
    """
    Analyze a single email for phishing indicators.

    Supports both raw EML format and pre-parsed email data.
    Uses XGBoost, Transformer, or Ensemble model based on model_type parameter.
    """
    email_id = _generate_email_id()
    start_time = datetime.now()

    try:
        # Check if feature extraction is available
        if not feature_extraction_service.is_available:
            raise HTTPException(
                status_code=503,
                detail="Feature extraction service not available. Ensure Day 1 pipeline is installed."
            )

        # Extract features
        if request.raw_email:
            extraction_result = await feature_extraction_service.extract_from_raw_email(
                request.raw_email
            )
            parsed_email = extraction_result["parsed_email"]
            features = extraction_result["features"]["features"]  # Extract actual features dict
        elif request.parsed_email:
            parsed_email = request.parsed_email
            extraction_result = await feature_extraction_service.extract_from_parsed_email(
                parsed_email
            )
            features = extraction_result["features"]["features"]  # Extract actual features dict
        else:
            raise HTTPException(status_code=400, detail="No email data provided")

        # Check cache for predictions
        cache_key = None
        if request.use_cache and request.model_type != ModelType.ENSEMBLE:
            # Generate cache key from features hash
            import hashlib
            import json
            features_str = json.dumps(features, sort_keys=True)
            cache_key = f"prediction:{request.model_type.value}:{hashlib.md5(features_str.encode()).hexdigest()}"
            cached_result = await cache_service.get(cache_key)

            if cached_result:
                logger.info(f"Prediction cache hit for {request.model_type.value}")
                record_cache_hit("model_prediction")
                return AnalysisResponse(**cached_result)

            record_cache_miss("model_prediction")

        # Run prediction based on model type
        model_type = request.model_type
        individual_predictions = []

        if model_type == ModelType.XGBOOST:
            if not settings.XGBOOST_AVAILABLE:
                raise HTTPException(status_code=503, detail="XGBoost model not available")

            prediction = predict_with_xgboost(features)
            individual_predictions.append(prediction)

        elif model_type == ModelType.TRANSFORMER:
            if not settings.TRANSFORMER_AVAILABLE:
                raise HTTPException(status_code=503, detail="Transformer model not available")

            prediction = predict_with_transformer(parsed_email)
            individual_predictions.append(prediction)

        elif model_type == ModelType.MULTI_AGENT:
            if not settings.MULTI_AGENT_AVAILABLE:
                raise HTTPException(status_code=503, detail="Multi-agent model not available")

            prediction = predict_with_multi_agent(parsed_email)
            individual_predictions.append(prediction)

        elif model_type == ModelType.ENSEMBLE:
            # Ensemble: use all available models
            all_predictions = []

            if settings.XGBOOST_AVAILABLE:
                xgb_pred = predict_with_xgboost(features)
                all_predictions.append(xgb_pred)

            if settings.TRANSFORMER_AVAILABLE:
                trans_pred = predict_with_transformer(parsed_email)
                all_predictions.append(trans_pred)

            if settings.MULTI_AGENT_AVAILABLE:
                ma_pred = predict_with_multi_agent(parsed_email)
                all_predictions.append(ma_pred)

            if len(all_predictions) == 0:
                raise HTTPException(status_code=503, detail="No models available for ensemble")

            individual_predictions = all_predictions

            # Calculate weighted ensemble score
            phishing_score = 0.0
            total_weight = 0.0

            weights = settings.get_ensemble_weights()

            for pred in all_predictions:
                model_name = pred.get("model_name", "unknown")
                weight = weights.get(model_name, 1.0 / len(all_predictions))

                if pred["verdict"] == "PHISHING":
                    phishing_score += pred["confidence"] * weight
                else:
                    phishing_score += (1 - pred["confidence"]) * weight

                total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                phishing_score /= total_weight

            # Determine ensemble verdict
            if phishing_score >= 0.6:
                verdict = "PHISHING"
                confidence = phishing_score
                risk_score = int(phishing_score * 100)
            elif phishing_score >= 0.4:
                verdict = "SUSPICIOUS"
                confidence = phishing_score + 0.1
                risk_score = int(phishing_score * 80)
            else:
                verdict = "LEGITIMATE"
                confidence = 1 - phishing_score
                risk_score = int(phishing_score * 30)

            prediction = {
                "model_name": "ensemble",
                "verdict": verdict,
                "confidence": confidence,
                "risk_score": risk_score,
                "individual_predictions": individual_predictions
            }

        # Extract prediction results
        verdict_str = prediction.get("verdict", "SUSPICIOUS")
        confidence = prediction.get("confidence", 0.5)
        risk_score = prediction.get("risk_score", 50)

        # Calculate processing time
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Build analysis breakdown
        analysis = {
            "feature_count": len(features),
            "url_features": {k: v for k, v in features.items() if k.startswith("url_")},
            "content_features": {k: v for k, v in features.items() if "urgency" in k or "threat" in k or "financial" in k},
            "financial_indicators": {k: v for k, v in features.items() if "bank" in k or "wire" in k or "credential" in k},
            "individual_predictions": individual_predictions if len(individual_predictions) > 1 else None
        }

        # Generate explanation
        if verdict_str == "PHISHING":
            explanation = f"High confidence phishing detection by {prediction.get('model_name', model_type.value)} model. Multiple risk factors detected."
        elif verdict_str == "SUSPICIOUS":
            explanation = f"Some phishing indicators detected by {prediction.get('model_name', model_type.value)} model. Manual review recommended."
        else:
            explanation = f"Email appears legitimate based on {prediction.get('model_name', model_type.value)} analysis."

        # Build response
        response = AnalysisResponse(
            email_id=email_id,
            verdict=Verdict(verdict_str),
            confidence=float(confidence),
            risk_score=risk_score,
            risk_level=_risk_score_to_level(risk_score),
            model_used=prediction.get("model_name", model_type.value),
            individual_predictions=individual_predictions if len(individual_predictions) > 1 else None,
            analysis=analysis,
            explanation=explanation,
            processing_time_ms=processing_time_ms,
            cache_hit=False,
            timestamp=datetime.utcnow().isoformat()
        )

        # Cache result
        if request.use_cache and cache_key:
            await cache_service.set(
                cache_key,
                response.model_dump(),
                ttl=settings.REDIS_PREDICTION_CACHE_TTL
            )

        # Record metrics
        record_model_prediction(
            model_type=model_type.value,
            verdict=verdict_str,
            duration_ms=processing_time_ms
        )

        logger.info(f"Email analyzed: {verdict_str} (confidence: {confidence:.2f}, risk: {risk_score})")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Email analysis failed: {str(e)}")


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple emails in batch.

    Maximum 100 emails per batch. Supports parallel processing.

    **Note**: Currently only URL analysis is fully implemented.
    Email analysis requires ML models (Phase 3).
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    start_time = datetime.now()

    try:
        # Validate batch size
        if len(request.emails) > 100:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of 100. Got {len(request.emails)} emails"
            )

        # Process emails in parallel if requested
        if request.parallel:
            tasks = []
            for email_req in request.emails:
                # Check if this is a URL-based request (dict with 'url' key)
                if isinstance(email_req, dict) and 'url' in email_req:
                    tasks.append(analyze_url(
                        URLAnalysisRequest(
                            url=email_req.get('url'),
                            context=email_req.get('context'),
                            use_cache=email_req.get('use_cache', True)
                        )
                    ))
                # Skip email analysis for now
                else:
                    logger.warning(f"Batch email analysis not yet supported, skipping: {type(email_req)}")

            results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        else:
            results = []
            for email_req in request.emails:
                if isinstance(email_req, dict) and 'url' in email_req:
                    try:
                        result = await analyze_url(
                            URLAnalysisRequest(
                                url=email_req.get('url'),
                                context=email_req.get('context'),
                                use_cache=email_req.get('use_cache', True)
                            )
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch item failed: {e}")
                else:
                    logger.warning(f"Batch email analysis not yet supported, skipping: {type(email_req)}")

        successful = [r for r in results if isinstance(r, AnalysisResponse)]
        failed = len(results) - len(successful)

        # Calculate summary statistics
        if successful:
            phishing_count = sum(1 for r in successful if r.verdict == Verdict.PHISHING)
            legitimate_count = sum(1 for r in successful if r.verdict == Verdict.LEGITIMATE)
            suspicious_count = sum(1 for r in successful if r.verdict == Verdict.SUSPICIOUS)
            avg_risk_score = sum(r.risk_score for r in successful) / len(successful)
            avg_confidence = sum(r.confidence for r in successful) / len(successful)
        else:
            phishing_count = legitimate_count = suspicious_count = 0
            avg_risk_score = avg_confidence = 0.0

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return BatchAnalysisResponse(
            batch_id=batch_id,
            results=successful,
            summary={
                "total_emails": len(request.emails),
                "phishing_count": phishing_count,
                "legitimate_count": legitimate_count,
                "suspicious_count": suspicious_count,
                "avg_risk_score": round(avg_risk_score, 2),
                "avg_confidence": round(avg_confidence, 2)
            },
            total_processing_time_ms=total_time,
            successful_count=len(successful),
            failed_count=failed
        )

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")
